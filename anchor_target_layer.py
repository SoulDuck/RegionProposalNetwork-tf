# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:11:17 2017
@author: Kevin Liang (modifications)
Anchor Target Layer: Creates all the anchors in the final convolutional feature
map, assigns anchors to ground truth boxes, and applies labels of "objectness"
Adapted from the official Faster R-CNN repo:
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
"""

# --------------------------------------------------------
# Faster R-CNN
# Written by KimSeongJung
# --------------------------------------------------------
import sys
#sys.path.append('../')

import numpy as np
import numpy.random as npr
import tensorflow as tf
import bbox_overlaps
import bbox_transform
import generate_anchor

def anchor_target(rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales):

    """
    rpn_cls_score 과 im_dims는 다르다
    Python version
    이해가 안가는게 ... 가끔 겹치지 않는 sample들이 있는데 그게 불가능 한데...뭐지....
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    """
    im_dims = im_dims[0]

    # _anchors shape : ( 9, 4 ) anchor coordinate type : x1,y1,x2,y2
    _anchors = generate_anchor.generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # Only minibatch of 1 supported
    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'
    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    # 1. Generate proposals from bbox deltas and shifted anchors
    # 1. 축소된 roi 형태를 원상 복귀 시칸다
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose() # 4,88 을 88,4 로 바꾼다

    A = _num_anchors # 9
    K = shifts.shape[0] # 88
    all_anchors=np.array([])
    for i in range(len(_anchors)):
        if i ==0 :
            all_anchors=np.add(shifts , _anchors[i])
        else:
            all_anchors = np.concatenate((all_anchors, np.add(shifts, _anchors[i])), axis=0)

    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    #print all_anchors
    # Element the useless coordinate
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_dims[1] + _allowed_border) &  # <-- width
        (all_anchors[:, 3] < im_dims[0] + _allowed_border))[0] # <-- height


    # 필요 있는 anchor 들만 남기고 나머지 제거
    anchors = all_anchors[inds_inside]
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # gt boxes 와 anchors 에서 얼마나 겹치는 지 확인한다
    # overlaps shape : [inds_inside , 2]
    # 만약 이미지안에 데이터가 하나만 있으면 당연히 [inds_inside , 1] 이다
    overlaps = bbox_overlaps.bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    """
    # 조금이라도 겹치는 indices
    overlay_indices=np.where([overlaps > 0])[1]
    print overlaps[overlay_indices]
    """

    # 여러 Ground Truth중에 가장 많이 겹치는 GroundTruth을 가져온다
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # inds_inside 갯수 만큼 overlaps에서 가장 높은 overlays

    # 모든 overlaps 중에서 가장 많이 겹치는 것을 가져온다
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # 가장 많이 겹치는 overlab 의 overlap 비율을 가져온다, [ 0.63126253  0.76097561]
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])] #*

    # 가장 많이 겹치는 overlab 의 arg 을 가져온다
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    #
    RPN_CLOBBER_POSITIVES = False
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION =0.5
    RPN_BATCHSIZE=60
    RPN_BBOX_INSIDE_WEIGHTS=(1.0, 1.0, 1.0, 1.0)
    RPN_POSITIVE_WEIGHT = -1.0

    if not RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1 # 가장 높은 anchor의 라벨은 1로 준다

    # fg label: above threshold IOU
    labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

    if RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # Training Set 에서  foreground 와 background 의 비율을 맞춘다
    num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE) # RPN_FG_FRACTION = 0.5
    fg_inds = np.where(labels == 1)[0]

    """
    print gt_boxes
    for gt in gt_boxes:
        x1, y1, x2, y2, l = gt
        print x2 - x1, y2 - y1
    """
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False) # replace = False --> 겹치지 않게 한다
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]

    if len(bg_inds) > len(fg_inds):
        disable_inds = npr.choice(bg_inds , size=(len(bg_inds) - len(fg_inds)) , replace=False)
        labels[disable_inds] = -1
    else:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    assert np.sum([labels ==0]) == np.sum([labels==1]) ,'{} {}'.format(np.sum([labels ==0]) , np.sum([labels ==1]))
    """
    print 'the number of all lables : ', np.shape(all_anchors)
    print 'the number of inside labels : ', np.shape(anchors)
    print 'the number of positive labels :', np.sum(labels == 1), '(anchor_target_layer.py)'
    print 'bg inds', np.sum([labels ==0])
    print 'fg inds', np.sum([labels == 1])
    # fg 는 무조건 하나 포함되는데 그 이유는 max IOU을 가지고 있는건 무조건 FG로 보게 한다
    # bg or fg 가 지정한 갯수보다 많으면 -1 라벨해서 선택되지 않게 한다
    # bbox_targets: The deltas (relative to anchors) that Faster R-CNN should
    # try to predict at each anchor
    # TODO: This "weights" business might be deprecated. Requires investigation
    """


    # 여기서부터는
    #bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32) 이게 왜 필요하지
    #Regression 을 할수 있게 변형한다

    # inside index 에 해당하는 gt boxes 들을 변환 시킨다.
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :]) #  bbox_targets = dx , dy , dw , dh


    # 나중에 사용할 bbox inside weight 와 outside weight 을 만든다
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels==1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS) #(1.0, 1.0, 1.0, 1.0)

    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if RPN_POSITIVE_WEIGHT < 0: #TRAIN.RPN_POSITIVE_WEIGHT = -1
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) # get positive label
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        #print 'positive weight ',positive_weights
        #print 'negative weight ',negative_weights
    else:
        assert ((RPN_POSITIVE_WEIGHT > 0) & (RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
        negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # deleteme
    """
    for i,box in enumerate(bbox_inside_weights):
        if np.sum(box) > 0:
            print i
    print ''
    for i,box in enumerate(bbox_outside_weights):
        if np.sum(box) > 0:
            print i
    print ''
    """
    """
    여기에서는 부적적한 coordinate 을 제거한다 
    anchor 의 coordinate 갯수와 label 의 갯수는 같다 
    label 이 뜻하는 것은 coordinate spot 에서의 bg 와 fg 을 뜻한다.
    만약 부적절한 coordinate spot 이라면 -1 을 지정한다 
    """
    # 전체 anchor 에 맞는 label 로 변환한다 (6283 --> 9900)
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    """
    for i,box in enumerate(bbox_inside_weights):
        if np.sum(box) > 0:
            print i
    print ''
    for i,box in enumerate(bbox_outside_weights):
        if np.sum(box) > 0:
            print i
    print ''
    """

    # label을 변환한다
    ori_labels = labels.reshape((1, height, width, A))
    labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2) # A = 9 , (1,h,w,9) ==> (1,9,h,w)
    labels = labels.reshape((1, 1, A * height, width)) # 왜 변하지 ?
    rpn_labels = labels


    # bbox_targets
    rpn_bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    # bbox_inside_weights
    rpn_bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    # bbox_outside_weights
    rpn_bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights ,bbox_targets ,bbox_inside_weights,bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4 # anchor
    assert gt_rois.shape[1] == 5 # gt_bbox

    return bbox_transform.bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
