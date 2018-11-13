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
def embedding(type, lo_input, hi_input, layer_nfilt, mult):
    dim = Config().basic_dim
    act = Config().basic_activation
    # do the lo-res input
    sel = layer_nfilt[0]
    lo = deconv2d(lo_input, sel, dim, act, 'emb_lo_{}_conv_{}'.format(type, sel))
    # do the hi-res input
    hi = kl.Conv2D(sel, dim, padding="same", activation=act,
                   name='emb_hi_{}_conv_{}'.format(type, sel))(hi_input)
    both = kl.Add()([lo,hi])
    sel = layer_nfilt[1]
    # TODO: compute this from the configuration or layer data!
    out = deconv2d(both, sel, dim, act, 'emb_comb_{}_conv_{}'.format(type, sel),
                   size=(mult,mult))
    return out

# build segmentation network
@check_tensor
def build_segmentation_network(input_, embed_, nfilts):
    dim = Config().basic_dim
    act = Config().basic_activation
    num_classes = Config().num_classes
    l = kl.Concatenate(axis=3)([input_,embed_])
    for i,nfilt in enumerate(nfilts):
        l = kl.Conv2D(nfilt, (dim,dim), activation=act, padding='same',
                      name='sem_seg_{}_{}'.format(nfilt,i))(l)
    l = kl.Conv2D(16, (dim,dim), activation='sigmoid', padding='same',
                  name='sem_seg_out_conv_sig1')(l)
    l = kl.Conv2D(num_classes, (1,1), activation='sigmoid', padding='same',
                  name='sem_seg_out_conv_sig2')(l)
    # threshold the mask
    l = kl.Lambda(lambda l: tf.to_float(l > 0.5), name="sem_seg_threshold")(l)
    l = kl.Permute((3,1,2),name='sem_seg_out')(l)
    return l

# extract a bounding box from the segmentation masks
@check_tensor
def build_bboxes_from_segmentation(masks_):
    # the masks are binary floats 0/1
    # so.... we can create a width range and a height range
    # and then element-wise multiply to find the min/max width and height
    # BxCxHxW
    H = masks_.shape[2]
    W = masks_.shape[3]
    wrange = tf.range(W)
    hrange = tf.range(H)
    X, Y = tf.meshgrid(wrange,hrange)
    xs = tf.reshape([1,1,H,W])
    ys = tf.reshape([1,1,H,W])
    bx = kl.Lambda(lambda masks_: masks_ * xs)(masks_)
    by = kl.Lambda(lambda masks_: masks_ * ys)(masks_)
    xmin = kl.Lambda(lambda bx: tf.reduce_min(bx,axis=[2,3]))(bx)
    xmax = kl.Lambda(lambda bx: tf.reduce_max(bx,axis=[2,3]))(bx)
    ymin = kl.Lambda(lambda by: tf.reduce_min(by,axis=[2,3]))(by)
    ymax = kl.Lambda(lambda by: tf.reduce_max(by,axis=[2,3]))(by)
    bboxes = kl.Concatenate(axis=2)([ymin,xmin,ymax,xmax])
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
def build_pose_rot_net(bboxes, probs, lo_input, hi_input, rot_layers):
    cfg = Config()
    dim = cfg.basic_dim
    act = cfg.basic_activation
    mask_thresh = cfg.mask_prob_threshold
    
    roi_size = Config().roi_size
    # given the *part* masks, we want to extract the full effective mask for the object
    # this means we want to filter by detection score
    # probs is NxNUM_CLASSES
    best_class = tf.argmax(probs, axis=1) # N
    best_prob = tf.reduce_max(probs, axis=1)
    active_masks = best_prob > mask_thresh # yields bool mask
    # collect the masks to create the object bounds
    
    
    print("build pose rot net: rois {}".format(rois.shape))
    roi_align = roi.RegionOfInterest(extent=(roi_size,roi_size))
    lo_pooled = roi_align([lo_input,rois])
    hi_pooled = roi_align([hi_input,rois])
    print("  roi align lo {}".format(lo_pooled.shape))
    print("  roi align hi {}".format(hi_pooled.shape))
    added = kl.Add()([lo_pooled,hi_pooled])
    print("  roi added {}".format(added.shape))
    l = kl.Flatten()(added)
    
    for i,sz in enumerate(rot_layers):
        l = kl.Dense(sz, activation=act, name=build_name('poserot_fc_{}'.format(sz),i))(l)
    out = kl.Dense(4,name='pose_rot_out')(l)
    print("pose rot out: {}".format(out.shape))
    LOG.debug("pose_rot_out: {}".format(out.shape))
    return out

@check_tensor
def build_part_net(image, mask, features, parts, trans_, rot_):
    cfg = Config()
    dim = cfg.basic_dim
    act = cfg.basic_activation
    input_dim = cfg.num_input_channels
    ds_stride = cfg.part_net.downscale_stride
    
    fl = features # at 1/16 scale
    ml = mask # at full scale
    il = image # at full scale
    pl = parts # at full scale
    
    # upsample the features to 1/4 scale
    fl = kl.UpSampling2D(name="pn_expand_feats_up", size=cfg.part_net.feature_dilation)(fl)
    fl = kl.Conv2DTranspose(cfg.part_net.feature_nfilt,
                            cfg.part_net.feature_dim,
                            dilation_rate=cfg.part_net.feature_dilation,
                            activation=act, padding='same', name='pn_expand_feats')(fl)    
    # downsample the full scale mask/image/parts to 1/4 scale
    ml = kl.Conv2D(1, cfg.part_net.downscale_dim, activation=act,
                   strides=(ds_stride,ds_stride), padding='same',
                       name='pn_contract_mask')(ml)
    il = kl.Conv2D(input_dim, cfg.part_net.downscale_dim, activation=act,
                   strides=(ds_stride,ds_stride), padding='same',
                       name='pn_contract_img')(il)

    #B = tf.shape(pl)[0]
    P = pl.shape[1]
    H = pl.shape[2]
    W = pl.shape[3]
    # handle the unsupported 5th dimension
    pl = kl.Reshape((P*H,W,2))(pl)
    dimdiv = 4
    pl = kl.AveragePooling2D(dimdiv, padding='same', name='pn_contract_parts')(pl)
    pl = kl.Reshape((P,H//dimdiv,W//dimdiv,2))(pl)

    # compute each part input (masked by part masks)
    # pl BxPxHxWx2 -> BxHxWxP (mask) BxHxWxP (depth) ] SPLIT ]
    # ml BxHxW     -> BxHxWx1
    # il BxHxWx4   -> BxHxWx4 (no change)
    # fl BxHxWxL   -> BxHxWxL (no change)
    epm = kl.Permute([2,3,1])(kl.Lambda(lambda pl: pl[...,0])(pl))
    epd = kl.Permute([2,3,1])(kl.Lambda(lambda pl: pl[...,1])(pl))
    efl = fl
    eil = il

    # NOW, stack all the inputs together
    mega_input = concat([epm, epd, efl, eil], axis=3) # should be P+P+4+L
    
    print("Parts il: {}".format(il.shape))
    print("Parts fl: {}".format(fl.shape))
    print("Mega input: {}".format(mega_input.shape))

    # create the part networks (using masked image, masked features, and the estimated depths)
    parts_out = []
    mega_in = mega_input
    for p,nfilt in enumerate(cfg.part_net.layers[0]):
        mega_in = kl.Conv2D(nfilt, dim, activation=act, padding='same',
                            name=build_name('mega_{}_{}'.format(p,nfilt)))(mega_in)
    mega_in = kl.Dense(cfg.part_net.fc_dim, activation=act)(mega_in)
    comb = kl.Flatten()(mega_in)

    all_parts = kl.Dense(cfg.num_primitive_parameters * cfg.num_primitives, activation=act)(comb)
    print("all parts: {}".format(all_parts.shape))

    assert_history("all_parts", all_parts)
    parts = kl.Reshape((cfg.num_primitives, cfg.num_primitive_parameters), name="parts")(all_parts)
    print("parts type: {}".format(type(parts)))
    
    # transform the parts to the camera frame
    assert_history("parts", parts)
    assert_history("trans_", trans_)
    assert_history("rot_", rot_)
    transformed_parts = transform_parts_layer(name="transformed_parts")([parts, trans_, rot_])
    return transformed_parts, parts
