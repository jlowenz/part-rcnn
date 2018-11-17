import ray
import pdb
import os
import time
from datetime import datetime
import numpy as np
import numpy.random as npr
import imgaug
import skimage
import cv2
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import keras.backend as K

from config import Config
import utils
import model as modellib
import visualize as vis
from utils import timeit, Timed

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
    DETECTION_NMS_THRESHOLD = 0.25
    
    # Training
    GPU_COUNT = 1
    # on the P6Ks, we can get at least 8
    IMAGES_PER_GPU = 4 # try it out, we have smaller images

    GRADIENT_CLIP_NORM = 10.0

    USE_MINI_MASK = False

    LEARNING_RATE = 0.001 # was 0.001, mask-rcnn sets 0.02
    SCHEDULE_FACTOR = 0.95 # was 0.9
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.00001 # was 0.0001

    PN_CONFIG = None

    @classmethod
    def init_from_pnconfig(cls, cfg):
        # initialize the ChairConfig from the partnet configuration so
        # we can use the file config convenience, and to keep
        # everything in conceptually one place
        cls.PN_CONFIG = cfg
        
        cls.NUM_CLASSES = cfg.num_classes
        cls.IMAGE_MIN_DIM = cfg.image_min_dim
        cls.IMAGE_MAX_DIM = cfg.image_max_dim

        cls.RPN_TRAIN_ANCHORS_PER_IMAGE = cfg.rpn_train_anchors_per_image
        cls.RPN_ANCHOR_SCALES = cfg.rpn_anchor_scales
        cls.TRAIN_ROIS_PER_IMAGE = cfg.train_rois_per_image

        # Use RGB-D data
        cls.USE_DEPTH = cfg.use_depth
        cls.USE_NORMALS = cfg.use_normals
        cls.SQUISH_INPUTS = cfg.squish_inputs

        cls.DETECTION_MIN_CONFIDENCE = cfg.detection_min_confidence
        cls.DETECTION_NMS_THRESHOLD = cfg.detection_nms_threshold

        # Training
        cls.GPU_COUNT = cfg.train.gpu_count
        # on the P6Ks, we can get at least 8
        cls.IMAGES_PER_GPU = cfg.train.images_per_gpu # try it out, we have smaller images

        cls.GRADIENT_CLIP_NORM = cfg.train.gradient_clip_norm

        cls.USE_MINI_MASK = cfg.train.use_mini_mask

        cls.LEARNING_RATE = cfg.train.learning_rate
        cls.SCHEDULE_FACTOR = cfg.train.schedule_factor 
        cls.LEARNING_MOMENTUM = cfg.train.learning_momentum
        cls.WEIGHT_DECAY = cfg.train.weight_decay
        cls.STEPS_PER_EPOCH = cfg.train.steps_per_epoch
    

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
                    "path": frm['rgb'],
                    "frame_id": pu.frame_num(frm['rgb']),
                    "depth": frm['depth'],
                    "instance": frm['instance'],
                    "normals": frm['normals'],
                    "masks": frm['part_masks'],
                    "pose": frm['pose'],
                    "prims": frm['prims']}
            self.image_info.append(info)
            i += 1
        print("Loaded {} chairs".format(len(self.image_info)))

    def pad(self, img):
        if img.ndim == 2:
            padding = [self.cfg_.vert_padding, self.cfg_.horiz_padding]
        else:
            padding = [self.cfg_.vert_padding, self.cfg_.horiz_padding, [0,0]]
        return np.pad(img, padding, mode='constant')

    @timeit
    def load_image(self, image_id):
        rgb = super(ChairDataset,self).load_image(image_id)
        return self.pad(rgb)

    @timeit
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

    @timeit
    def load_depth(self, image_id):
        frm = self.image_info[image_id]
        depth = cv2.imread(str(frm['depth']), cv2.IMREAD_UNCHANGED)
        if depth.ndim != 3:
            depth = np.expand_dims(depth, axis=2)
        return self.pad(depth)

    @timeit
    def load_normals(self, image_id):
        cfg = self.cfg_
        #depth = self.load_depth(image_id)
        normals = np.load(self.image_info[image_id]['normals'])
        assert normals.dtype == np.float32, "Incorrect normals type {}".format(normals.dtype)
        assert np.amax(normals) <= 1.0 and np.amin(normals) >= -1
        return self.pad(normals)

    @timeit
    def load_gt_seg(self, image_id):
        #inst_id = self.image_info[image_id]['instance_id']
        seg = cv2.imread(str(self.image_info[image_id]['instance']),cv2.IMREAD_UNCHANGED)
        ids = np.sort(np.unique(seg))
        inst_id = ids[-1] # instance IDs are broken.... but last ID *should* work
        img = np.zeros(seg.shape, dtype=np.float32)
        img[seg == inst_id] = 1.0
        if np.amax(img) != 1.0:
            print("----------------------- Bogus object mask found")
            print("image id:    {}".format(image_id))
            print("instance id: {}".format(inst_id))
            print("ids in img : {}".format(ids))
            print("Info: {}".format(self.image_info[image_id]))
        return self.pad(img)

    @timeit
    def load_primitives(self, image_id):
        prims_file = self.image_info[image_id]['prims']
        d = np.load(str(prims_file))
        return d['primitives']

    def load_pose(self, image_id):
        return self.image_info[image_id]['pose']

def parse_epoch(path):
    parts = str(path).split("_")
    for p in parts:
        if p.startswith("ep"):
            epoch = int(p[2:])
            return epoch
    return 0

def get_latest_model(model_dir, tag):
    models = model_dir.glob("*.h5")
    tagged_models = [(m.stat().st_mtime,m) for m in models if m.name.startswith(tag)]

    print("Tagged models:\n",tagged_models)
    if len(tagged_models) > 0:
        tagged_models.sort(key=lambda x: x[0], reverse=True)
        latest_model = tagged_models[0][1]
        parts = str(latest_model).split("_")
        epoch = parse_epoch(parts)
        print("Found recent model at epoch {}: {}".format(
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
def filter_results(cfg, cnames, r):
    rois = r['rois'] # Nx4
    ids = r['class_ids'] # N
    scores = r['scores'] # N
    masks = r['masks'] # HxWxN

    out_rois = []
    out_ids = []
    out_scores = []
    out_masks = []
    detection_filter = cfg.detection_filter.to_dict()
    for i in range(rois.shape[0]):
        part = cnames[ids[i]]
        if part in detection_filter: 
            thresh = detection_filter[part]
            if scores[i] < thresh:
                continue
        out_rois.append(rois[i])
        out_ids.append(ids[i])
        out_scores.append(scores[i])
        out_masks.append(masks[:,:,i])
    if len(out_rois) < 1:
        # everything filtered
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    out_rois = np.stack(out_rois,axis=0)
    out_ids = np.stack(out_ids).astype(np.int32)
    out_scores = np.stack(out_scores)
    out_masks = np.stack(out_masks,axis=2)
    return out_rois, out_masks, out_ids, out_scores

def evaluate(cfg, dataset, model_file):
    # TODO: turn the following into a shared function    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.device(cfg.device):
        print("Constructing session on {}".format(cfg.device))
        session = tfdbg.LocalCLIDebugWrapperSession(tf.Session(config=config))
        K.set_session(session)
    
    # create the config
    chair_cfg = ChairConfig()

    # create the model
    model = modellib.MaskRCNN(mode="inference", config=chair_cfg,
                              model_dir=str(cfg.paths.model_dir))

    # load the weights file
    loaded = False
    if cfg.use_previous_model:
        epoch, model_path = get_latest_model(cfg.paths.model_dir, cfg.tag)
        if model_path:
            print("Loading weights: {}".format(model_path))
            model.load_weights(str(model_path), by_name=True)
            loaded = True

    if not loaded and (model_file or cfg.paths.model_weights):
        model_path = model_file or cfg.paths.model_weights
        if model_path is not None:
            model.load_weights(str(model_path), by_name=True)

    train,val,test = generate_sample_splits(cfg, "chair")
    
    # load the train and val datasets
    chair_val = ChairDataset(cfg, dataset, val)

    # create image output directory
    now = datetime.now().isoformat()
    eval_path = pl.Path(str(cfg.paths.eval_path).format(tstamp=now))
    if not eval_path.exists():
        eval_path.mkdir(parents=True, exist_ok=True)

    # write some metadata to the directory
    metadata = eval_path / "metadata.txt"
    with open(str(metadata),"w") as file:
        file.write("model: {}".format(model.loaded_weight_file_))
        file.write("epoch: {}".format(model.epoch))
    
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
            rois, masks, ids, scores = filter_results(cfg, class_names, r)
            if ids.size > 0:
                vis.display_instances(images[i][0], rois, masks, ids,
                                      class_names, scores, save=True,
                                      base_dir=str(eval_path),
                                      image_id=b*chair_cfg.BATCH_SIZE+i)


################################################################################
# Training
################################################################################

def train(cfg, dataset, model_file=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.device(cfg.device):
        print("Constructing session on {}".format(cfg.device))
        #session = tfdbg.LocalCLIDebugWrapperSession(tf.Session(config=config))
        session = tf.Session(config=config)
        K.set_session(session)
    
    # create the config
    chair_cfg = ChairConfig()

    # create the model
    model = modellib.MaskRCNN(mode="training", config=chair_cfg,
                              model_dir=str(cfg.paths.model_dir))

    # imagenet init
    if cfg.init_with_imagenet:
        w = model.get_imagenet_weights()
        model.load_weights(w, by_name=True, exclude=['conv1'])
    
    # load the weights file
    loaded = False
    if cfg.use_previous_model:
        epoch, model_path = get_latest_model(cfg.paths.model_dir, cfg.tag)
        if model_path:
            print("Loading weights: {}".format(model_path))
            model.load_weights(str(model_path), by_name=True)
            loaded = True

    if not loaded and (model_file or cfg.paths.model_weights):
        model_path = model_file or cfg.paths.model_weights
        if model_path is not None:
            model.load_weights(str(model_path), by_name=True)

                
    train,val,test = generate_sample_splits(cfg, "chair")
    
    # load the train and val datasets
    chair_train = ChairDataset(cfg, dataset, train)
    chair_val = ChairDataset(cfg, dataset, val)

    # image augmentation
    augmentation = None#imgaug.augmenters.Fliplr(0.5)


    print("Starting at model epoch: {}".format(model.epoch))
    with tf.device(cfg.device):
        # Training schedule
        learning_rate = chair_cfg.LEARNING_RATE
        while model.epoch < 50:
            print("TRAINING all")
            model.train(chair_train, chair_val,
                        learning_rate = learning_rate,
                        total_epochs = 50,
                        layers = 'all',
                        augmentation=augmentation)
        # while model.epoch < 50:
        #     print("TRAINING 4+")
        #     model.train(chair_train, chair_val,
        #                 learning_rate = learning_rate,
        #                 num_epochs = 1,
        #                 layers = '4+',
        #                 augmentation=augmentation)
        #     print("TRAINING all")
        #     model.train(chair_train, chair_val,
        #                 learning_rate = learning_rate,
        #                 num_epochs = 1,
        #                 layers = 'all',
        #                 augmentation=augmentation)
        #     learning_rate *= chair_cfg.SCHEDULE_FACTOR
        # learning_rate /= 2.0
        # while model.epoch < 100:
        #     model.train(chair_train, chair_val,
        #                 learning_rate = learning_rate,
        #                 num_epochs = 1,
        #                 layers = 'heads',
        #                 augmentation=augmentation)
        #     model.train(chair_train, chair_val,
        #                 learning_rate = learning_rate,
        #                 num_epochs = 1,
        #                 layers = 'all',
        #                 augmentation=augmentation)
        #     learning_rate *= chair_cfg.SCHEDULE_FACTOR
        # learning_rate = chair_cfg.LEARNING_RATE
        # while model.epoch < 160:
        #     # print("TRAINING 4+")
        #     # model.train(chair_train, chair_val,
        #     #             learning_rate = learning_rate,
        #     #             num_epochs = 1,
        #     #             layers = '4+',
        #     #             augmentation=augmentation)
        #     print("TRAINING ALL")
        #     model.train(chair_train, chair_val,
        #                 learning_rate = learning_rate,
        #                 total_epochs = 160,
        #                 layers = 'all',
        #                 augmentation=augmentation)
        #     learning_rate *= chair_cfg.SCHEDULE_FACTOR


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
    parser.add_argument('-m', '--model-file', type=str,
                        help="Specify a model file to load")

    args = parser.parse_args()

    cfg = PNConfig(pl.Path(args.config_file))
    ChairConfig.init_from_pnconfig(cfg)

    #ray.init()
    dataset = pd.Dataset(args.data_dirs, filter_key='chair')
    #ray.shutdown()
    cfg.dataset = dataset
    
    if args.command == 'train':
        train(cfg,dataset,args.model_file)
    elif args.command == 'evaluate':
        evaluate(cfg,dataset,args.model_file)
    else:
        print("'{}' is not a recognized command")
        parser.print_help()

if __name__ == "__main__":
    main()
