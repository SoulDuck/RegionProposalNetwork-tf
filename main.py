import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes
from anchor_target_layer import anchor_target
from convnet import define_placeholder , simple_convnet , rpn_cls_layer , rpn_bbox_layer , sess_start , optimizer ,rpn_cls_loss , rpn_bbox_loss
import math
rpn_labels_op = tf.placeholder(dtype =tf.int32 , shape=[1,1,None,None])
rpn_bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])

bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])
bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])
bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])



x_, im_dims, gt_boxes, phase_train = define_placeholder()
top_conv, _feat_stride = simple_convnet(x_)

# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)
# CLS LOSS
rpn_cls_loss_op ,A_op ,B_op = rpn_cls_loss(rpn_cls , rpn_labels_op)

# BBOX LOSS
# C_op : indiced rpn bbox  pred op
# D_op :indiced rpn target  op
# E_op : rpn_inside_weights
# F_op : indices
rpn_bbox_loss_op , diff_op , C_op , D_op ,E_op ,F_op = \
    rpn_bbox_loss(rpn_bbox_pred ,bbox_targets_op , bbox_inside_weights_op , bbox_outside_weights_op , rpn_labels_op)



cost_op = rpn_bbox_loss_op + rpn_cls_loss_op
rpn_cls_train_op = optimizer(rpn_cls_loss_op)
rpn_bbox_train_op = optimizer(rpn_bbox_loss_op )
train_op = optimizer(rpn_bbox_loss_op )

sess=sess_start()
for i in range(2,55000 * 100):

    src_img , src_gt_boxes =next_img_gtboxes(i)
    h,w=np.shape(src_img)
    src_im_dims = [(h,w)]
    anchor_scales = [3,4,5]
    rpn_cls_score=np.zeros([1,int(math.ceil(h/8.)),int(math.ceil(w/8.)),512])
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
        rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=8, anchor_scales=anchor_scales)

    """
    rpn_labels : (1, 1, h*9 , w)
    rpn_bbox_targets : (1, 4*9, h, w)
    rpn_bbox_inside_weights : (1, 4*9,  h, w)
    rpn_bbox_outside_weights: (1, 4*9,  h, w)
    """

    """    
    # delete me
    n, ch, h, w = np.shape(rpn_bbox_targets)
    rpn_bbox_targets = np.reshape(rpn_bbox_targets , [1,4, h*9 , w])
    rpn_bbox_targets = np.transpose(rpn_bbox_targets , [0,2,3,1])
    rpn_bbox_targets = np.reshape(rpn_bbox_targets, [-1 , 4])
    for rpn_bbox in rpn_bbox_targets:
        print rpn_bbox
    """

    # utils here
    src_img=src_img.reshape([1]+list(np.shape(src_img))+[1])
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels,
                 rpn_bbox_targets_op: rpn_bbox_targets ,
                 rpn_bbox_inside_weights_op: rpn_bbox_inside_weights ,
                 rpn_bbox_outside_weights_op : rpn_bbox_outside_weights
                 }
    cost, _ ,rpn_cls_value, A, B, diff, C, D ,E ,F= sess.run(
        fetches=[rpn_cls_loss_op, rpn_cls_train_op, rpn_cls, A_op, B_op, diff_op, C_op, D_op ,E_op ,F_op], feed_dict=feed_dict)
    print '#### indices ####'
    print F
    print '#### cos ####'
    print cost
    print '#### pred ####'
    print C
    print '#### target ####'
    print D
    print '#### C -D ####'
    print C-D
    print '#### diff ####'
    print diff
    print '#### E ####'
    print E
    for i in range(len(F)):
        print F[i][0]
        print E[F[i][0]]
    for i,e in enumerate(E):
        if np.sum(e) > 0:
            print i
            print e
    exit()


