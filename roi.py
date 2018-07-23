import tensorflow as tf
import proposal_layer
def _rpn_softmax(rpn_cls_layer):
    shape = tf.shape(rpn_cls_layer)
    rpn_cls_score = tf.transpose(rpn_cls_layer, [0, 3, 1, 2])  # Tensor("transpose:0", shape=(1, 18, ?, ?)
    rpn_cls_score = tf.reshape(rpn_cls_score, [shape[0], 2, shape[3] // 2 * shape[1],
                                               shape[2]])  # Tensor("transpose:0", shape=(1, 2, ?, ?)
    rpn_cls_score = tf.transpose(rpn_cls_score,
                                 [0, 2, 3, 1])  # Tensor("transpose_1:0", shape=(?, ?, ?, 2), dtype=float32)

    rpn_cls_prob_ori = tf.nn.softmax(
        rpn_cls_score)  # Tensor("transpose_1:0", shape=(?, ?, ?, 2), dtype=float32)
    # Reshape back to the original
    rpn_cls_prob = tf.transpose(rpn_cls_prob_ori,
                                [0, 3, 1, 2])  # Tensor("transpose_2:0", shape=(?, 2, ?, ?), dtype=float32)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [shape[0], shape[3], shape[1], shape[
        2]])  # Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)
    rpn_cls_prob = tf.transpose(rpn_cls_prob,
                                [0, 2, 3, 1])  # Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)

    return rpn_cls_prob



def roi_proposal(rpn_cls_layer , rpn_bbox_layer, im_dims , _feat_stride , anchor_scales , is_training):
    print '########################################################'
    print '########## ROI Proposal Network building.... ###########'
    print '########################################################'

    num_classes = 10 + 1  # 1 -> background

    rpn_cls_prob = _rpn_softmax(rpn_cls_layer)
    blobs, scores = proposal_layer.proposal_layer(rpn_bbox_cls_prob=rpn_cls_prob,
                                                            rpn_bbox_pred=rpn_bbox_layer,
                                                            im_dims=im_dims, cfg_key=is_training,
                                                            _feat_stride=_feat_stride,
                                                            anchor_scales=anchor_scales)
    return blobs , scores
    """
    if is_training:  # train
        # Calculate targets for proposals
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer.proposal_target_layer(rpn_rois=blobs, gt_boxes=gt_boxes,
                                                        _num_classes=num_classes)
    """

