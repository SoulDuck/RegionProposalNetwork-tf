# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:11:17 2017
@author: Kevin Liang (modifications)
generate_anchors and supporting functions: generate reference windows (anchors)
for Faster R-CNN. Specifically, it creates a set of k (default of 9) relative
coordinates. These references will be added on to all positions of the final
convolutional feature maps.
Adapted from the official Faster R-CNN repo:
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py
Note: the produced anchors have indices off by 1 of what the comments claim.
Probably due to MATLAB being 1-indexed, while Python is 0-indexed.
"""

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=2, ratios=[0.5, 1, 2],scales=2**np.arange(3, 6)):

    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 #[ 0,0,15,15] ==> x1 , y1, x2 , y2
    ratio_anchors = _ratio_enum(base_anchor, ratios) # 0,0 , 15 ,15 #ratios , ratio_anchors shape = (3,4)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) #scales =  8 ,18 ,32 , _scale_enum = []
                         for i in range(ratio_anchors.shape[0])]) #ratio_anchors.shape[0] = 3

    """
    print 'anchros scales : {}'.format(scales)
    print 'anchors width , height '
    for a in anchors:
        x1,y1,x2,y2=a
        print x2-x1 ,y2-y1
    """
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # 0,0 , 15 ,15 #ratios
    w = anchor[2] - anchor[0] + 1 # anchor[2]  = 15 ,anchor[0]  = 15
    h = anchor[3] - anchor[1] + 1 # anchor[3]  = 15 ,anchor[1]  = 15
    x_ctr = anchor[0] + 0.5 * (w - 1) #
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))


    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor) #
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios)) # ws [ 23.  16.  11.]
    hs = np.round(ws * ratios) # hs [ 12.  16.  22.]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # scales = [ 8 ,16 ,32]
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales #scales
    hs = h * scales

    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)


