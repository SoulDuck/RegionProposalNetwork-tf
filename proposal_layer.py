#-*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
from bbox_transform import bbox_transform_inv, clip_boxes
from configure import cfg
import generate_anchor
from nms_wrapper import nms


def proposal_layer(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, cfg_key, _feat_stride, anchor_scales):
    blobs , scores =tf.py_func(_proposal_layer_py,
                                 [rpn_bbox_cls_prob, rpn_bbox_pred, im_dims[0], cfg_key, _feat_stride, anchor_scales],
                                 [tf.float32 , tf.float32 ])
    blobs=tf.reshape(blobs , shape=[-1,5])
    scores=tf.reshape(scores , shape=[-1])
    return blobs ,  scores

def _proposal_layer_py(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, cfg_key, _feat_stride, anchor_scales):
    '''
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    '''
    _anchors = generate_anchor.generate_anchors(scales=np.array(anchor_scales)) # #_anchors ( 9, 4 )
    _num_anchors = _anchors.shape[0] #9
    rpn_bbox_cls_prob = np.transpose(rpn_bbox_cls_prob, [0, 3, 1, 2]) # rpn bbox _cls prob # 1, 18 , h , w
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2]) # 1, 36 , h , w
    # Only minibatch of 1 supported
    assert rpn_bbox_cls_prob.shape[0] == 1, \
        'Only single item batches are supported'
    if cfg_key == 'TRAIN':
        pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N #12000
        post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N # 2000
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH #0.7
        min_size = cfg.TRAIN.RPN_MIN_SIZE # 16

    else:  # cfg_key == 'TEST':
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        min_size = cfg.TEST.RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs


    scores = rpn_bbox_cls_prob[:,  _num_anchors : , :, :] # 1, 18  , H, W --> 1, 9, H, W
    bbox_deltas = rpn_bbox_pred

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # Enumerate all shifted anchors:
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]

    #anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = np.array([])
    for i in range(len(_anchors)):
        if i == 0:
            anchors = np.add(shifts, _anchors[i])
        else:
            anchors = np.concatenate((anchors, np.add(shifts, _anchors[i])), axis=0)
    anchors = anchors.reshape((K * A, 4))
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1)) # (h * w * A , 1)
    # anchors ,bbox_deltas , scores 모두 같은 shape 여야 한다

    proposals = bbox_transform_inv(anchors, bbox_deltas)
    proposals = clip_boxes(proposals, im_dims) # image size 보다 큰 proposals 들이 줄어 들수 있도록 한다.
    keep = _filter_boxes(proposals, min_size) # min size = 16 # min보다 큰 놈들만 살아남았다

    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    #print 'scores : ',np.shape(scores) #421 ,13 <--여기 13이 자꾸 바귄다..
    order = scores.ravel().argsort()[::-1] # 크기 순서를 뒤집는다 가장 큰 값이 먼저 오게 한다

    if pre_nms_topN > 0: #120000
        order = order[:pre_nms_topN]
    #print np.sum([scores>0.7])

    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    #print np.shape(np.hstack ((proposals , scores))) # --> [x_start , y_start ,x_end, y_end , score ] 이런 형태로 만든다
    keep = nms(np.hstack((proposals, scores)), nms_thresh) # nms_thresh = 0.7 | hstack --> axis =1
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    #print post_nms_topN

    proposals = proposals[keep, :]
    scores = scores[keep]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False))) # N , 5

    #blob=np.hstack((blob , scores))

    return blob , scores

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _inv_transform_layer_py(rpn_bbox_pred,  is_training, _feat_stride, anchor_scales , indices):
    _anchors = generate_anchor.generate_anchors(scales=np.array(anchor_scales)) # #_anchors ( 9, 4 )
    _num_anchors = _anchors.shape[0] #9
    shape = np.shape(rpn_bbox_pred)
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2]) # 1, 36 , h , w
    rpn_bbox_pred = np.reshape(rpn_bbox_pred, [shape[0], 4, shape[3] // 4 * shape[1], shape[2]])  # 1, 4 , h * 9 , w
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, (0, 2, 3, 1)) # 1, h * 9 , w , 4
    bbox_deltas = rpn_bbox_pred
    bbox_deltas = bbox_deltas.reshape((-1, 4))

    if is_training == 'TRAIN':
        pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N #12000
        post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N # 2000
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH #0.7
        min_size = cfg.TRAIN.RPN_MIN_SIZE # 16
    else:  # cfg_key == 'TEST':
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        min_size = cfg.TEST.RPN_MIN_SIZE
    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = shape[1] , shape[2]
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # Enumerate all shifted anchors:
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    #anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))

    anchors = np.array([])
    for i in range(len(_anchors)):
        if i == 0:
            anchors = np.add(shifts, _anchors[i])
        else:
            anchors = np.concatenate((anchors, np.add(shifts, _anchors[i])), axis=0)
    anchors = anchors.reshape((K * A, 4))

    # anchors ,bbox_deltas , scores 모두 같은 shape 여야 한다
    proposals = bbox_transform_inv(anchors, bbox_deltas)
    #proposals = clip_boxes(proposals, im_dims) # image size 보다 큰 proposals 들이 줄어 들수 있도록 한다.
    target_proposals=proposals[indices]
    return proposals , target_proposals

def inv_transform_layer(rpn_bbox_pred, cfg_key, _feat_stride, anchor_scales , indices):

    proposals, target_proposals = tf.py_func(_inv_transform_layer_py, [rpn_bbox_pred, cfg_key, _feat_stride, anchor_scales , indices],
                       [tf.float32 , tf.float32])
    proposals=tf.reshape(proposals , shape=[-1,4])
    target_proposals = tf.reshape(target_proposals, shape=[-1, 4])
    return proposals, target_proposals

