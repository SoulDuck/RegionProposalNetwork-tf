import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME' , act=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k,k,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        if not act is None:
            layer=act(layer , name='relu')
        else:
            print 'activation None'
        if __debug__ == True:
            print 'layer name : ' ,name
            print 'layer shape : ' ,layer.get_shape()

        return layer
def convolution2d_manual(name,x,out_ch,k_h ,k_w , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k_h,k_w,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        layer=tf.nn.relu(layer , name='relu')
        if __debug__ == True:
            print 'layer name : ', name
            print 'layer shape : ' ,layer.get_shape()

    return layer
def max_pool(name,x , k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        if __debug__ ==True:

            layer=tf.nn.max_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)
            print 'layer name :', name
            print 'layer shape :', layer.get_shape()
    return layer
def avg_pool(name,x , k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        if __debug__ ==True:

            layer=tf.nn.avg_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)
            print 'layer name :', name
            print 'layer shape :', layer.get_shape()
    return layer

def batch_norm_layer(x,phase_train,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=scope_bn)
    z = tf.cond(phase_train, lambda: bn_train, lambda: bn_inference)
    return z
def affine(name,x,out_ch , trainable=True , activation=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc = tf.get_variable('w', [height * width * in_ch, out_ch],
                                   initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc = tf.get_variable('w', [in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=trainable)

        b_fc = tf.Variable(tf.constant(0.1), out_ch,  name='b')
        layer=tf.matmul(x , w_fc) + b_fc
        if not activation is None:
            layer=activation(layer)
        else:
            print 'Activation is None'

        print 'layer name : {}'.format(name)
        print 'layer shape :',layer.get_shape()
        return layer
def logits(name,x, n_classes ):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc = tf.get_variable('w', [height * width * in_ch, n_classes],
                                   initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc = tf.get_variable('w', [in_ch, n_classes], initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)
        logits= tf.matmul(x, w_fc)
        return logits


def gap(name,x , n_classes ):
    in_ch=x.get_shape()[-1]
    gap_x=tf.reduce_mean(x, (1,2) ,name='global_average_pooling')
    with tf.variable_scope(name) as scope:
        gap_w=tf.get_variable('w' , shape=[in_ch , n_classes] , initializer=tf.random_normal_initializer(0,0.01) , trainable=True)
    y_conv=tf.matmul(gap_x, gap_w , name='logits')
    return y_conv

def lr_schedule(step ,lr_iters , lr_values):
    assert len(lr_iters) == len(lr_values)

    def _fn(step, lr_iters, lr_values):
        n_lr_iters = len(lr_iters)
        for idx in range(n_lr_iters):
            if step < lr_iters[idx]:
                return lr_iters[idx], lr_values[idx]
            elif idx <= n_lr_iters - 1:
                continue
        return lr_iters[idx], lr_values[idx]
    lr_iter , lr_value=_fn(step , lr_iters ,lr_values)
    return lr_value

def dropout(x_ , phase_train , keep_prob):
    print 'dropout applied'
    return tf.cond(phase_train , lambda : tf.nn.dropout(x_ , keep_prob=keep_prob) , lambda: x_)

def l2_loss(optimizer ,loss_tensor ):

        print 'l2 loss'
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(loss_tensor + l2_loss * weight_decay, name='train_op')
        return train_op


def algorithm(y_conv , y_ , learning_rate , optimizer , use_l2_loss):

    assert int(y_conv.get_shape()[-1]) == int(y_.get_shape()[-1]) \
        , 'logits : {} true labels :{}'.format(y_conv.get_shape()[-1] , y_.get_shape()[-1])
    """

    :param y_conv: logits
    :param y_: labels
    :param learning_rate: learning rate
    :return:  pred,pred_cls , cost , correct_pred ,accuracy

    """
    if __debug__ ==True:
        print 'debug start : cnn.py | algorithm'
        print 'optimizer option : sgd(default) | adam | momentum | '
        print 'selected optimizer : ',optimizer
        print y_conv.get_shape()
        print y_.get_shape()
    optimizer_dic = {'sgd': tf.train.GradientDescentOptimizer(learning_rate), 'adam': tf.train.AdamOptimizer(learning_rate),
                     'momentum': tf.train.MomentumOptimizer(learning_rate , momentum=0.9 , use_nesterov=True)}

    pred=tf.nn.softmax(y_conv , name='softmax')
    pred_cls=tf.argmax(pred , axis=1 , name='pred_cls')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv , labels=y_) , name='cost')

    if use_l2_loss:
        train_op=l2_loss(optimizer_dic[optimizer], cost)
    else:
        train_op = optimizer_dic[optimizer].minimize(cost,name='train_op')
    correct_pred=tf.equal(tf.argmax(y_conv , 1) , tf.argmax(y_ , 1) , name='correct_pred')
    accuracy =  tf.reduce_mean(tf.cast(correct_pred , dtype=tf.float32) , name='accuracy')
    return pred,pred_cls , cost , train_op,correct_pred ,accuracy




if __name__ == '__main__':
    print 'a'
