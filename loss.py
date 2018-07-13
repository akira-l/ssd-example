import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_tensor, labels=target_tensor)
    prediction_tensor = tf.convert_to_tensor(tf.cast(prediction_tensor, tf.float32), tf.float32)
    target_tensor = tf.convert_to_tensor(tf.cast(target_tensor, tf.float32), tf.float32)
    zeros = array_ops.zeros_like(target_tensor, dtype=target_tensor.dtype)
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - softmax_cross_entropy, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, softmax_cross_entropy)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(softmax_cross_entropy, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - softmax_cross_entropy, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
    
    
# smooth_L1 
def smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))
