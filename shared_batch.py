import numpy as np
import multiprocessing as mp
import ctypes
import pdb

def zeros(shape, dtype=np.float32):
    z = np.zeros(shape,dtype=dtype)
    a = mp.RawArray(ctypes.c_byte, z.nbytes) # we will manage the synchro
    b = np.frombuffer(a, dtype=dtype)
    b = b.reshape(shape)
    b[:] = z # initialize the data, since the other way is NOT WORKING!
    return a, shape, dtype

class SharedBatch(object):
    def __init__(self, idx, N, queue_size, proto, cfg, anchors,
                 random_rois, detection_targets):
        self.index = idx
        self.step = 0
        self.N = N
        self.queue_size = queue_size
        self.config = cfg
        self.anchors = anchors
        self.random_rois = random_rois
        self.detection_targets = detection_targets
        self.batch_size = cfg.BATCH_SIZE
        self.qoffset = idx * self.batch_size
        self.read_complete = mp.Condition()
        self.ready_for_read = mp.Value(ctypes.c_bool)
        self.ready_for_read.value = False

        i = self.step * self.queue_size * self.batch_size + self.qoffset
        self.item_index = mp.Value(ctypes.c_int, i)
        print("creating shared batch {}".format(self.index))
        self.create_(proto)
        print("done shared batch {}".format(self.index))
        self.enable_write()

    def create_(self, proto):
        cfg = self.config.PN_CONFIG
        self.batch_image_meta_ = zeros(
            (self.batch_size,) + proto.image_meta.shape, dtype=proto.image_meta.dtype)
        self.batch_rpn_match_ = zeros(
            [self.batch_size, self.anchors.shape[0], 1], dtype=proto.rpn_match.dtype)
        self.batch_rpn_bbox_ = zeros(
            [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
            dtype=proto.rpn_bbox.dtype)
        self.batch_images_ = zeros(
            (self.batch_size,) + proto.image.shape, dtype=np.float32)
        if self.config.USE_DEPTH:
            self.batch_depth_ = zeros((self.batch_size,) + proto.depth.shape,
                                      dtype=np.float32)
        if self.config.USE_NORMALS:
            self.batch_normals_ = zeros((self.batch_size,) + proto.normals.shape,
                                          dtype=np.float32)
        if cfg.enable_segmentation_extension:
            self.batch_gt_seg_ = zeros((self.batch_size,1,) + proto.gt_seg.shape,
                                         dtype=np.float32)
        if cfg.enable_primitive_extension:
            self.batch_gt_pose_  = zeros((self.batch_size,7), dtype=np.float32)
            self.batch_gt_prims_ = zeros((self.batch_size,cfg.num_primitives,
                                          cfg.num_parameters), dtype=np.float32)
        self.batch_gt_class_ids_ = zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
        self.batch_gt_boxes_ = zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        if self.config.USE_MINI_MASK:
            self.batch_gt_masks_ = zeros((self.batch_size, self.config.MINI_MASK_SHAPE[0],
                                       self.config.MINI_MASK_SHAPE[1],
                                       self.config.MAX_GT_INSTANCES))
        else:
            self.batch_gt_masks_ = zeros(
                (self.batch_size, proto.image.shape[0], proto.image.shape[1],
                 self.config.MAX_GT_INSTANCES))
        if self.random_rois:
            self.batch_rpn_rois_ = zeros(
                (self.batch_size, proto.rpn_rois.shape[0], 4),
                dtype=proto.rpn_rois.dtype)
            if self.detection_targets:
                self.batch_rois_ = zeros(
                    (self.batch_size,) + proto.rois.shape, dtype=proto.rois.dtype)
                self.batch_mrcnn_class_ids_ = zeros(
                    (self.batch_size,) + proto.mrcnn_class_ids.shape,
                    dtype=proto.mrcnn_class_ids.dtype)
                self.batch_mrcnn_bbox_ = zeros(
                    (self.batch_size,) + proto.mrcnn_bbox.shape,
                    dtype=proto.mrcnn_bbox.dtype)
                self.batch_mrcnn_mask_ = zeros(
                    (self.batch_size,) + proto.mrcnn_mask.shape,
                    dtype=proto.mrcnn_mask.dtype)

    def tonp(self, a):
        arr, shape, dtype = a
        b = np.frombuffer(arr,dtype=dtype)
        return b.reshape(shape)
                
    @property
    def batch_image_meta(self):
        return self.tonp(self.batch_image_meta_)

    @property
    def batch_rpn_match(self):
        return self.tonp(self.batch_rpn_match_)

    @property
    def batch_rpn_bbox(self):
        return self.tonp(self.batch_rpn_bbox_)

    @property
    def batch_images(self):
        return self.tonp(self.batch_images_)

    @property
    def batch_depth(self):
        return self.tonp(self.batch_depth_)

    @property
    def batch_normals(self):
        return self.tonp(self.batch_normals_)

    @property
    def batch_gt_seg(self):
        return self.tonp(self.batch_gt_seg_)

    @property
    def batch_gt_pose(self):
        return self.tonp(self.batch_gt_pose_)

    @property
    def batch_gt_prims(self):
        return self.tonp(self.batch_gt_prims_)
    
    @property
    def batch_gt_class_ids(self):
        return self.tonp(self.batch_gt_class_ids_)

    @property
    def batch_gt_boxes(self):
        return self.tonp(self.batch_gt_boxes_)

    @property
    def batch_gt_masks(self):
        return self.tonp(self.batch_gt_masks_)

    @property
    def batch_rpn_rois(self):
        return self.tonp(self.batch_rpn_rois_)

    @property
    def batch_rois(self):
        return self.tonp(self.batch_rois_)

    @property
    def batch_mrcnn_class_ids(self):
        return self.tonp(self.batch_mrnn_class_ids_)

    @property
    def batch_mrcnn_bbox(self):
        return self.tonp(self.batch_mrcnn_bbox_)

    @property
    def batch_mrcnn_mask(self):
        return self.tonp(self.batch_mrcnn_mask_)    

    def __item_index(self):
        return self.step * self.queue_size * self.batch_size + self.qoffset
    
    def next_step(self):
        self.step += 1
        i = self.__item_index()
        if i > self.N:
            self.step = 0
            i = self.__item_index()
        self.item_index = i  
                    
    def enable_read(self):
        with self.read_complete:
            self.ready_for_read.value = True
            self.read_complete.notify()

    def can_write(self):
        return not self.ready_for_read.value
    def can_read(self):
        return self.ready_for_read.value
                
    def enable_write(self):
        with self.read_complete:
            self.next_step()
            self.ready_for_read.value = False
            self.read_complete.notify()
                     
