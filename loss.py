import tensorflow as tf

def Custom_L1(y_true, y_pred):
    Err = y_true - y_pred
    Abs = tf.abs(Err)
    Mean = tf.reduce_mean(Abs, axis = [1, 2, 3])
    return Mean

def Custom_MSE(y_true, y_pred):
    Err = y_true - y_pred
    Square = tf.square(Err)
    Mean = tf.reduce_mean(Square, axis = [1, 2, 3])
    return Mean

def Custom_RMSE(y_true, y_pred):
    Err = y_true - y_pred
    Square = tf.square(Err)
    Mean = tf.sqrt(tf.reduce_mean(Square, axis = [1, 2, 3]))
    return Mean

def Custom_SSIM(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    # b, h, w, c = tf.shape(y_true)
    #print(tf.shape(y_true))

    # [b, h, w, c] -> [b*c, h, w]
    tmp_true = tf.reshape(tf.transpose(y_true, [0, 3, 1, 2]),
                          [tf.shape(y_true)[0]*tf.shape(y_true)[3], tf.shape(y_true)[1], tf.shape(y_true)[2], 1])
    tmp_pred = tf.reshape(tf.transpose(y_pred, [0, 3, 1, 2]),
                          [tf.shape(y_true)[0]*tf.shape(y_true)[3], tf.shape(y_true)[1], tf.shape(y_true)[2], 1])
    max_val = tf.reduce_max(tmp_true, axis=(1, 2, 3), keepdims=True)
    
    ssim = tf.image.ssim(tmp_true/max_val, tmp_pred/max_val, max_val=1.)
    
    # ssim : [b*c, 1] -> [b, c]
    ssim = tf.clip_by_value(tf.reduce_mean(tf.reshape(ssim, [tf.shape(y_true)[0], tf.shape(y_true)[3]]), axis=1), 0, 1)
    
    return (1.-ssim)/2.

class multi_loss():
    def __init__(self, a, b, type):
        self.a = a
        self.b = b
        self.type = type
    def loss(self, y_true, y_pred):
        if self.type == 'l1ssim':
            loss1 = Custom_L1(y_true, y_pred)
            loss2 = Custom_SSIM(y_true, y_pred)
            return self.a*loss1 + self.b*loss2
        
        if self.type == 'l2ssim':
            loss1 = Custom_L2(y_true, y_pred)
            loss2 = Custom_SSIM(y_true, y_pred)
            return self.a*loss1 + self.b*loss2
        
def Custom_L1_SSIM(y_true, y_pred):
    loss1 = Custom_L1(y_true, y_pred)
    loss2 = Custom_SSIM(y_true, y_pred)
    return 95*loss1 + 5*loss2

            
    
def Custom_L2_SSIM(y_true, y_pred):
    loss1 = Custom_MSE(y_true, y_pred)
    loss2 = Custom_SSIM(y_true, y_pred)
    return 95*loss1 + 5*loss2
    
def mutual_information_single(hist2d):
    tmp = tf.cast(hist2d, dtype='float64')
    pxy = tmp / tf.reduce_sum(tmp)
    px = tf.reduce_sum(pxy, axis=1)
    py = tf.reduce_sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = tf.greater(pxy, 0)
    return tf.reduce_sum(tf.boolean_mask(pxy, nzs) * tf.log(tf.boolean_mask(pxy, nzs) / tf.boolean_mask(px_py, nzs)))

def tf_joint_histogram(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    #print("joint1")
    vmax = 255
    #b, h, w, c = tf.shape(y_true)
    
    
    # Intensity Scaling
    max_int = tf.reduce_max(y_true, axis = [1,2], keepdims=True)
    tmp_true = tf.round(y_true / max_int * vmax)
    tmp_pred = tf.round(y_pred / max_int * vmax)
    
    #print("joint2")
    # [batch, height, width, channel]
    # -> [batch, height * width, channel]
    # -> [batch, channel, height * width]
    flat_true = tf.transpose(tf.reshape(tmp_true,
                                        [tf.shape(y_true)[0], tf.shape(y_true)[1]*tf.shape(y_true)[2], tf.shape(y_true)[-1]]), [0, 2, 1])
    flat_true = tf.reshape(flat_true, [tf.shape(y_true)[0]*tf.shape(y_true)[-1], tf.shape(y_true)[1]*tf.shape(y_true)[2]])
    flat_pred = tf.transpose(tf.reshape(tmp_pred, [tf.shape(y_true)[0], tf.shape(y_true)[1]*tf.shape(y_true)[2], tf.shape(y_true)[-1]]), [0, 2, 1])
    flat_pred = tf.reshape(flat_pred, [tf.shape(y_true)[0]*tf.shape(y_true)[-1], tf.shape(y_true)[1]*tf.shape(y_true)[2]])
    #print("joint3")
    output = (flat_pred * (vmax+1)) + (flat_true+1)
    #print("joint4")
    # [b*c, 65536]
    output = tf.map_fn(lambda x : tf.cast(tf.histogram_fixed_width(x, value_range=[1, (vmax+1)**2], nbins=(vmax+1)**2), 'float32'), output)
    # [b, c, 256, 256] -> [b, 256, 256, c]
    output = tf.transpose(tf.reshape(output, [tf.shape(y_true)[0], tf.shape(y_true)[-1], vmax+1, vmax+1]), [0, 2, 3, 1])
    #print("joint5")
    return output, y_true, y_pred

def mutual_information(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    # [b, 256, 256, c]
    joint_histogram, _, _ = tf_joint_histogram(y_true, y_pred)
    #b, h, w, c = tf.shape(joint_histogram)
    #print("mutual1")
    # [b*c, 256, 256]
    reshape_joint_histogram = tf.reshape(tf.transpose(joint_histogram, [0, 3, 1, 2]), [tf.shape(joint_histogram)[0]*tf.shape(joint_histogram)[-1], tf.shape(joint_histogram)[1], tf.shape(joint_histogram)[2]])
    #print("mutual2")
    output = tf.map_fn(lambda x : mutual_information_single(x), reshape_joint_histogram, dtype=tf.float64)
    #print("mutual3")
    output = tf.reshape(output, [tf.shape(joint_histogram)[0], tf.shape(joint_histogram)[-1]])
    return tf.cast(1 - tf.reduce_mean(output, axis=1), 'float32')