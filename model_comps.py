import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as kl
import keras.initializers as ki
import keras.engine as ke
import keras.models as km
import keras.regularizers as kr
import tfquaternion as tfq

from roi_align import PyramidROIAlign, PyramidROIAlign_v2

from partnet.config import Config
from partnet.util import build_name, assert_history

def check_tensor(f):
    def call_f(*args, **kwargs):
        outs = f(*args, **kwargs)
        assert_history(f, outs)
        return outs
    return call_f

@check_tensor
def deconv2d(input_, nfilt, dim, act, name, size=(2,2)):
    l = input_
    l = kl.UpSampling2D(name="{0}_up".format(name), size=size)(l)
    l = kl.Conv2D(nfilt, (dim,dim), activation=act, padding='same', name=name)(l)
    return l

@check_tensor
def embedding(type, roi_feats, layer_nfilt):
    dim = Config().basic_dim
    act = Config().basic_activation
    #roi_shape = roi_feats.shape
    x = roi_feats
    assert_history("roi_feats in embedding {}".format(type), x)
    print("embedded {} roi shape {}".format(type, x.shape))
    for i,filt in enumerate(layer_nfilt):
        # do the hi-res input
        x = kl.TimeDistributed(kl.Conv2D(filt, dim, padding="same", activation=act),
                                   name='emb_hi_{}_conv_{}'.format(type, i))(x)
    return x

# build segmentation network
@check_tensor
def build_segmentation_network(input_, embed_, nfilts):
    dim = Config().basic_dim
    act = Config().basic_activation
    num_classes = Config().num_object_classes 
    scale = int(input_.shape[-2]) // int(embed_.shape[-2])
    input_ = kl.MaxPooling2D(pool_size=scale)(input_)
    l = kl.Concatenate(axis=3)([input_,embed_])
    for i,nfilt in enumerate(nfilts):
        l = kl.Conv2D(nfilt, (dim,dim), activation=act, padding='same',
                      name='sem_seg_{}_{}'.format(nfilt,i))(l)
    l = kl.Conv2D(16, (dim,dim), activation='sigmoid', padding='same',
                  name='sem_seg_out_conv_sig1')(l)
    # TODO: this is NOT correct! True multi-instance detection would be more like the
    # rest of the RPN framework
    l = kl.Conv2D(num_classes, (1,1), activation='sigmoid', padding='same',
                  name='sem_seg_out_conv_sig2')(l)
    # threshold the mask
    l = kl.Lambda(lambda l: tf.round(l), name="sem_seg_threshold")(l)
    # scale up the mask
    l = kl.UpSampling2D(scale, interpolation='bilinear')(l)
    l = kl.Permute((3,1,2),name='sem_seg_out')(l)
    return l

# extract a bounding box from the segmentation masks
@check_tensor
def build_bboxes_from_segmentation(masks_):
    # the masks are binary floats 0/1
    # so.... we can create a width range and a height range
    # and then element-wise multiply to find the min/max width and height
    # BxCxHxW
    H = int(masks_.shape[2])
    W = int(masks_.shape[3])
    wrange = tf.range(W)
    hrange = tf.range(H)
    X, Y = tf.meshgrid(wrange,hrange)
    xs = tf.cast(tf.reshape(X,[1,1,H,W]), tf.float32)
    ys = tf.cast(tf.reshape(Y,[1,1,H,W]), tf.float32)
    bx = kl.Lambda(lambda masks_: masks_ * xs, name="mult_bx")(masks_)
    byy = kl.Lambda(lambda masks_: masks_ * ys, name="mult_by")(masks_)
    xmin = kl.Lambda(lambda bx: tf.expand_dims(tf.reduce_min(bx,axis=[2,3]),2),
                     name="minx")(bx)
    xmax = kl.Lambda(lambda bx: tf.expand_dims(tf.reduce_max(bx,axis=[2,3]),2),
                     name="maxx")(bx)
    ymin = kl.Lambda(lambda b: tf.expand_dims(tf.reduce_min(b,axis=[2,3]),2),
                     name="miny")(byy)
    ymax = kl.Lambda(lambda b: tf.expand_dims(tf.reduce_max(b,axis=[2,3]),2),
                     name="maxy")(byy)
    bboxes = kl.Concatenate(axis=2)([ymin,xmin,ymax,xmax])
    # need to normalize the bounding boxes
    normalizer = tf.constant([H,W,H,W],dtype=tf.float32)
    bboxes = kl.Lambda(lambda bb: bb / tf.reshape(normalizer, [1,1,4]),
                       name="normalize")(bboxes)
    return bboxes

# define the network for computing the pose translation
@check_tensor
def build_pose_trans_net(input_, trans_layers):
    dim = Config().basic_dim
    act = Config().basic_activation
    conv_layers, fc_layers = trans_layers
    l = input_
    for i,nfilt in enumerate(conv_layers):
        l = kl.Conv2D(nfilt, (dim,dim), activation=act, padding='same',
                      name=build_name('posetrans_{}'.format(nfilt), i))(l)
        l = kl.MaxPooling2D()(l)
    l = kl.Flatten()(l)
    for i,sz in enumerate(fc_layers):
        l = kl.Dense(sz,activation=act,
                     name=build_name('posetrans_fc_{}'.format(sz),i))(l)
    out = kl.Dense(3,name='pose_trans_out')(l)
    return out

# define the network for computing the pose rotation
@check_tensor
def build_pose_rot_net(input_, rot_layers):
    cfg = Config()
    dim = cfg.basic_dim
    act = cfg.basic_activation

    l = input_
    conv_layers, fc_layers = rot_layers
    # conv
    for i,nfilt in enumerate(conv_layers):
        l = kl.Conv2D(nfilt, (dim,dim), activation=act, padding='same',
                      name=build_name('poserot_{}'.format(nfilt),i))(l)
        
    for i,sz in enumerate(fc_layers):
        l = kl.Dense(sz, activation=act,
                     name=build_name('poserot_fc_{}'.format(sz),i))(l)
    l = kl.Flatten()(l)
    out = kl.Dense(4,name='pose_rot_out')(l)
    print("pose rot out: {}".format(out.shape))
    return out

def primitive_init(shape, dtype=None):
    print("primitive init / shape {}".format(shape))
    return K.random_normal(shape, dtype=dtype) * 3.0

@check_tensor
def build_part_net(bboxes, masks, features, trans_, rot_):
    '''
    Construct the part-primitive prediction network.

    Utilizes the detected parts (but they come as small, distorted masks... unless we convert them to something else)

    Outputs:
    tparts, parts, int_parts = build_part_net(...)

    where:

    tparts:    the array of transformed parts in camera frame (BxNxQ) where Q = |{sx,sy,sz,qw,qx,qy,qz,tx,ty,tz}|
    parts:     the array of parts in the canonical frame (BxNxQ)
    int_parts: the intermediate part parameters (BxNxI) where I = |{sx,sy,sz,qw,qx,qy,qz,ox,oy,depth}|
    '''
    cfg = Config()
    dim = cfg.basic_dim
    act = cfg.basic_activation

    # for EACH part (bbox,mask,feats), we want to compute the int_part output
    # this includes the primitive:
    #  - scale,
    #  - rotation (in camera frame),
    #  - x,y image offset from the CENTER of the bbox,
    #  - and depth in camera frame
    #
    # We can then use these values to compute the part parameters in both the canonical frame and the
    # camera frame

    # let's pass the features through several convolutional nets (for the halibut)
    fs = features
    for i,nfilt in enumerate(cfg.part_net.conv_layers):
        fs = kl.TimeDistributed(kl.Conv2D(nfilt, dim, padding="same", activation=act),
                                name="pn_conv_{}_{}".format(i,nfilt))(fs)
    # now we want to merge the features with the mask... HOW?
    # upsample the features to the size of the masks
    fs = kl.TimeDistributed(kl.UpSampling2D(cfg.part_net.feature_dilation, interpolation="bilinear"),
                            name="pn_up_{0}x{0}".format(cfg.part_net.feature_dilation))(fs)
    # concatenate the features and the masks
    # features (BxNxHxWxC)
    # masks    (BxNxHxWxP)
    # C : number of feature channels
    # P : number of part classes
    # how can these GO TOGETHER? We could generate a single map, or a 10 channel map PER class PER roi
    # COMPRESS to depth 10
    for i,nfilt in enumerate(cfg.part_net.merge_layers):
        fs = kl.TimeDistributed(kl.Conv2D(nfilt, dim, padding="same", activation=act),
                                name="pn_merge_{}_{}".format(i,nfilt))(fs)
    fs = kl.Multiply()([fs,masks])

    # fs is now BxNxHxWxP (one feature map channel per part class)
    # compute the intermediate part parameters
    # scale, rotation in camera frame, x-y offsets, depth
    # we are going for: BxNxPxQ
    # for each batch, for each roi, for each part class, primitive parameters
    x = fs
    x = kl.TimeDistributed(kl.Permute((3,1,2)),name="pn_permute_feats")(x)
    F = 28*28 # TODO: get this from the config     
    x = kl.TimeDistributed(kl.Reshape((-1,F)),name="pn_reshape_feats")(x)
    # Now, we have BxNxPxF
    for i,nfilt in enumerate(cfg.part_net.fc_layers):
        x = kl.TimeDistributed(kl.Dense(nfilt,activation=act),name="pn_fc_{}_{}".format(i,nfilt))(x)
    # now, final layer output to intermediate parts
    int_parts = kl.TimeDistributed(kl.Dense(cfg.part_net.num_params,bias_initializer=primitive_init),
                                   name="pn_int_parts")(x)

    # given the intermediate parts, the first step is to compute the transformed (camera) parts
    # maybe we should just call them "camera" parts?
    transformed_parts = kl.Lambda(lambda x: intermediate_parts_to_camera(*x))([int_parts, bboxes])
    parts = kl.Lambda(lambda p: transformed_parts_to_object(*p))([transformed_parts, trans_, rot_])
    
    return transformed_parts, parts, int_parts

def intermediate_parts_to_camera(int_parts, bboxes):
    cfg = Config()    
    # int_parts: BxNxPxQ
    # bboxes: BxNxS
    #
    # Q is {sx,sy,sz,qw,qx,qy,qz,ox,oy,depth}
    # S is {y1,x1,y2,x2} <-- normalized
    #
    # we need to compute the backprojected points for the translation
    # i.e. ox,oy,depth --> tx,ty,tz
    print("int parts shape {}".format(int_parts.shape))
    print("bboxes shape    {}".format(bboxes.shape))
    B = tf.shape(int_parts)[0]
    N = int_parts.shape[1]
    P = int_parts.shape[2]
    [fx,fy,cx,cy] = cfg.cam_params
    # first, compute the (x,y) centroid of the bounding boxes

    y = tf.reshape((bboxes[:,:,:,2] - bboxes[:,:,:,0]) / 2.0 * cfg.input_height,(B,N,P,1))
    x = tf.reshape((bboxes[:,:,:,3] - bboxes[:,:,:,1]) / 2.0 * cfg.input_width,(B,N,P,1))
    #y_gt_0 = tf.assert_greater_equal(y, 0.0, [y], name="y_greater_zero")
    #x_gt_0 = tf.assert_greater_equal(x, 0.0, [x], name="x_greater_zero")
    #with tf.control_dependencies([y_gt_0,x_gt_0]):
    ox = tf.expand_dims(int_parts[:,:,:,7],3)
    oy = tf.expand_dims(int_parts[:,:,:,8],3)
    z  = tf.expand_dims(int_parts[:,:,:,9],3)
    # fX/Z+cx
    # (x - cx)Z/f
    # add the offsets
    adj_x = x + ox
    adj_y = y + oy
    # backproject
    X = (adj_x - cx)*z/fx
    Y = (adj_y - cy)*z/fy
    Z = z
    tparts = tf.concat([int_parts[:,:,:,:7], X, Y, Z],axis=3)
    return tparts

def transformed_parts_to_object(tparts, trans, rot):
    # we have the translation and rotation that transforms the object to the camera frame
    # therefore, we need to compute the inverse of the transform in order to
    # translate the primitive definitions back to object pose
    # This will allow us to apply direct loss to the values, since we have the part
    #
    # tparts: BxNxPxQ
    uq = tf.reshape(rot, [-1,1,1,4]) # B,1,1,4
    t = tf.reshape(trans, [-1,1,1,3]) # B,1,1,3
    rq = tfq.Quaternion(uq)
    q = rq.normalized()
    invq = q.conjugate()
    invT = -tfq.rotate_vector_by_quaternion(invq,t)
    print("invT: {}".format(invT.shape))

    print("TPARTS: {}".format(tparts.shape))
    # transform the 
    ts = tparts[:,:,:,7:] # BxNxPx3
    uq = tfq.Quaternion(tparts[:,:,:,3:7]) # BxNxPx4
    qs = uq.normalized()
    print("qs shape {}".format(qs.value().shape))

    Tts = tfq.rotate_vector_by_quaternion(invq, ts) + invT
    Tqs = tfq.quaternion_multiply(qs,invq)
    print("Tqs   : {}".format(Tqs.shape))
    parts = tf.concat([tparts[:,:,:,:3],Tqs,Tts], axis=3)
    print("OPARTS: {}".format(parts.shape))
    return parts

    
    
