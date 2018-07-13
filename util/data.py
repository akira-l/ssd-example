import os 
import numpy as np
import tensorflow as tf
import random

class get_image(object):
    def decode_train_tfrecord(train_path, stage_):
        train_addr = os.path.join(train_path, stage_+str(bag_counter)+".tfrecords")
        filename_queue = tf.train.string_input_producer([train_addr])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, 
features={'img_raw':tf.FixedLenFeature([], tf.string), 'width':tf.FixedLenFeature([], tf.int64), 'height':tf.FixedLenFeature([], tf.int64)})
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.cast(img, tf.float32)*(1./255)-0.5
        return img
