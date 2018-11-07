import os
import time
import numpy as np
import numpy.random as npr
import imgaug
import skimage
import tensorflow as tf
import keras.backend as K

from config import Config
import utils
import model as modellib
import visualize as vis

import pathlib as pl

from partnet.config import Config as PNConfig
import partnet.dataset as pd
import partnet.util as pu
import partnet.normal as pn

class ChairConfig(Config):

    NAME = "chairs"

    # Number of part classes (including background)
    NUM_CLASSES = 1 + 13

    # image sizes (square), divisible by 2^6
    # whereas 240 is not
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    RPN_TRAIN_ANCHORS_PER_IMAGE = 49 # 5^2
    RPN_ANCHOR_SCALES = (16,32,64,128,160)
    TRAIN_ROIS_PER_IMAGE = 64
    
    # Use RGB-D data
    USE_DEPTH = True
    USE_NORMALS = True
    SQUISH_INPUTS = True

    DETECTION_MIN_CONFIDENCE = 0.65
    DETECTION_NMS_THRESHOLD = 0.2
    
    # Training
    GPU_COUNT = 1
    # on the P6Ks, we can get at least 8
    IMAGES_PER_GPU = 16 # try it out, we have smaller images

    GRADIENT_CLIP_NORM = 10.0

    USE_MINI_MASK = False

    LEARNING_RATE = 0.01 # was 0.001, mask-rcnn sets 0.02
    SCHEDULE_FACTOR = 0.95 # was 0.9
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.00001 # was 0.0001
    
    def __init__(self):
        super(ChairConfig,self).__init__()
    

################################################################################
# Chairs Dataset
################################################################################
class ChairDataset(utils.Dataset):
    def __init__(self, cfg, dataset, selection):
        super(ChairDataset,self).__init__()
        self.cfg_ = cfg
        self.dataset_ = dataset
        self.selection_ = selection
        # load the chair labels
        self.id_to_label_ = {}
        self.label_to_id_ = {}
        with open(str(self.cfg_.paths.label_path),"r") as f:
            for line in f.readlines():
                label, si = line.strip().split(" ")
                i = int(si)
                if i == 0:
                    continue # already added in base
                self.id_to_label_[i] = label
                self.label_to_id_[label] = i
                self.class_info.append({"source": "chairs",
                                        "id": i,
                                        "name": label})        
        self.load_chairs(self.selection_)
        self.prepare()
        assert(len(self.class_info) == 14)
    
    def load_chairs(self, select):
        samples = self.dataset_.samples_list(["chair"])
        sel_samples = [s for i,s in enumerate(samples) if i in select]
        
        i = 0        
        for mdl,frm in sel_samples:
            info = {"id": i,
                    "source": "chairs",
                    "path": frm[0],
                    "frame_id": pu.frame_num(frm[0]),
                    "depth": frm[1],
                    "instance": frm[2],
                    "masks": frm[4]}
            self.image_info.append(info)
            i += 1
        print("Loaded {} chairs".format(len(self.image_info)))

    def pad(self, img):
        padding = [self.cfg_.vert_padding, self.cfg_.horiz_padding, (0,0)]
        return np.pad(img, padding, mode='constant')

    def load_image(self, image_id):
        rgb = super(ChairDataset,self).load_image(image_id)
        return self.pad(rgb)
    
    def load_mask(self, image_id):
        frm = self.image_info[image_id]
        mask_path = frm['masks']
        m = np.load(str(mask_path))
        bboxes = m['bboxes']
        masks = m['masks']
        mask_labels = m['mask_labels']
        # handle a bug in the mask processing
        if masks.shape[0] != mask_labels.size:
            # bug exists if the labels don't match the # of masks
            masks = masks[:-1,:,:] # remove the empty end mask        
        # need to convert masks to binary/boolean
        bmasks = masks == 255
        bmasks = np.transpose(bmasks, axes=[1,2,0])
        # construct the class_ids array
        class_ids = mask_labels
        if np.any(class_ids > self.cfg_.num_primitives):
            raise RuntimeError("Frame: {} has broken class ids".format(frm))
        return self.pad(bmasks), class_ids

    def load_depth(self, image_id):
        frm = self.image_info[image_id]
        depth = skimage.io.imread(frm['depth'])
        if depth.ndim != 3:
            depth = np.expand_dims(depth, axis=2)
        return self.pad(depth)

    def load_normals(self, image_id):
        cfg = self.cfg_
        #depth = self.load_depth(image_id)
        depth = skimage.io.imread(self.image_info[image_id]['depth'])
        if depth.ndim != 3:
            depth = np.expand_dims(depth, axis=2)
        f = (cfg.cam_params[0] + cfg.cam_params[1])/2.0
        normals = np.zeros([depth.shape[0],depth.shape[1],3], dtype=np.float32)
        pn.depth_to_normals(depth[:,:,0], normals, f=f)
        return self.pad(normals)

def get_latest_model(model_dir, tag):
    models = model_dir.glob("*.h5")
    tagged_models = [(m.stat().st_mtime,m) for m in models if m.name.startswith(tag)]

    print("Tagged models:\n",tagged_models)
    if len(tagged_models) > 0:
        tagged_models.sort(key=lambda x: x[0], reverse=True)
        latest_model = tagged_models[0][1]
        parts = str(latest_model).split("_")
        epoch = parse_epoch(parts)
        LOG.info("Found recent model at epoch {}: {}".format(
            epoch,tagged_models[0][1]))
        return epoch, tagged_models[0][1]
    else:
        return 0, None

def generate_sample_splits(cfg, cat):
    items = cfg.dataset.cat_sample_list(cat)
    num_items = len(items)
    train_count = int(num_items * cfg.train.train_val_test_split[0])
    val_count = int(num_items * cfg.train.train_val_test_split[1])
    test_count = num_items - (train_count + val_count)
    # generate a permutation using the random generator
    perm = npr.permutation(num_items)
    return (perm[:train_count],
            perm[train_count:(train_count+val_count)],
            perm[(train_count+val_count):])
    
    
################################################################################
# Evaluation
################################################################################

def evaluate(cfg, dataset):
    # TODO: turn the following into a shared function    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.device(cfg.device):
        print("Constructing session on {}".format(cfg.device))
        session = tf.Session(config=config)
        K.set_session(session)
    
    # create the config
    chair_cfg = ChairConfig()

    # create the model
    model = modellib.MaskRCNN(mode="inference", config=chair_cfg,
                              model_dir=str(cfg.paths.model_dir))

    # load the weights file
    if cfg.use_previous_model:
        epoch, model_path = get_latest_model(cfg.paths.model_dir, cfg.tag)
        if model_path:
            print("Loading weights: {}".format(model_path))
            model.load_weights(str(model_path), by_name=True)
        else:
            model.load_weights(str(cfg.paths.model_weights), by_name=True, exclude=cfg.excluded_weights)
    else:
        model.load_weights(str(cfg.paths.model_weights), by_name=True, exclude=cfg.excluded_weights)
    
    train,val,test = generate_sample_splits(cfg, "chair")
    
    # load the train and val datasets
    chair_val = ChairDataset(cfg, dataset, val)

    #gen = modellib.DataGenerator(chair_val, chair_cfg, batch_size=4)
    image_ids = npr.permutation(chair_val.image_ids)
    total_batches = len(image_ids) // chair_cfg.BATCH_SIZE
    for b in range(total_batches // 2):
        images = []
        start = b * chair_cfg.BATCH_SIZE
        end = start + chair_cfg.BATCH_SIZE
        for i in image_ids[start:end]:
            img = [chair_val.load_image(i)]
            if chair_cfg.USE_DEPTH:            
                img.append(chair_val.load_depth(i))
            if chair_cfg.USE_NORMALS:            
                img.append(chair_val.load_normals(i))
            images.append(img)
        with tf.device(cfg.device):
            results = model.detect(images)
        # TODO: load the parts from the FILE
        class_names = ['unlabeled',
                       'base',
                       'support',
                       'seat',
                       'back',
                       'leftarm',
                       'rightarm',
                       'front left leg',
                       'front right leg',
                       'rear left leg',
                       'rear right leg',
                       'desktop',
                       'left rocker',
                       'right rocker']
        for i,r in enumerate(results):
            vis.display_instances(images[i][0], r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'], save=True, base_dir=cfg.paths.eval_path,
                                  image_id=b*chair_cfg.BATCH_SIZE+i)


################################################################################
# Training
################################################################################

def train(cfg, dataset):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.device(cfg.device):
        print("Constructing session on {}".format(cfg.device))
        session = tf.Session(config=config)
        K.set_session(session)
    
    # create the config
    chair_cfg = ChairConfig()

    # create the model
    model = modellib.MaskRCNN(mode="training", config=chair_cfg,
                              model_dir=str(cfg.paths.model_dir))

    # load the weights file
    if cfg.use_previous_model:
        epoch, model_path = get_latest_model(cfg.paths.model_dir, cfg.tag)
        if model_path:
            print("Loading weights: {}".format(model_path))
            model.load_weights(str(model_path), by_name=True)
        else:
            model.load_weights(str(cfg.paths.model_weights), by_name=True, exclude=cfg.excluded_weights)
    else:
        model.load_weights(str(cfg.paths.model_weights), by_name=True, exclude=cfg.excluded_weights)
    
    train,val,test = generate_sample_splits(cfg, "chair")
    
    # load the train and val datasets
    chair_train = ChairDataset(cfg, dataset, train)
    chair_val = ChairDataset(cfg, dataset, val)

    # image augmentation
    augmentation = imgaug.augmenters.Fliplr(0.5)


    with tf.device(cfg.device):
        # Training schedule
        print("Training all layers")
        if model.epoch < 100:
            model.train(chair_train, chair_val,
                        learning_rate = chair_cfg.LEARNING_RATE,
                        epochs = 100,
                        layers = 'all',
                        augmentation=augmentation)
        if model.epoch < 150:
            model.train(chair_train, chair_val,
                        learning_rate = chair_cfg.LEARNING_RATE / 2.0,
                        epochs = 70,
                        layers = '4+',
                        augmentation=augmentation)
        
                

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Train PartMask R-CNN on chairs dataset")
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Chairs Dataset")
    parser.add_argument('-d', '--data-dirs', type=str, action='append',
                        help="Data directories containing chair scenes")
    parser.add_argument('-f', '--config-file', type=str,
                        help="Configuration file")

    args = parser.parse_args()

    cfg = PNConfig(pl.Path(args.config_file))
    dataset = pd.Dataset(args.data_dirs, filter_key='chair')
    cfg.dataset = dataset
    
    if args.command == 'train':
        train(cfg,dataset)
    elif args.command == 'evaluate':
        evaluate(cfg,dataset)
    else:
        print("'{}' is not a recognized command")
        parser.print_help()

if __name__ == "__main__":
    main()
