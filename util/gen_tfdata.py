import os 
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

dataset_path = os.getcwd()+"/wtf"
out_path = os.getcwd()

def gen_tfrecord():
    writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "train.tfrecords"))
    for pic_name in range(1000):
        file_addr = os.path.join(dataset_path, str(pic_name)+".jpg")
        img = mpimg.imread(fname = file_addr)
        img_raw = img.tostring()
        anno = np.load("/home/liang/wider face/label.npy")
        anno_ = anno[pic_name]
        example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "xmin":tf.train.Feature(float_list=tf.train.FloatList(value=anno_[0])),
                    "ymin":tf.train.Feature(float_list=tf.train.FloatList(value=anno_[1])),
                    "xmax":tf.train.Feature(float_list=tf.train.FloatList(value=anno_[2])),
                    "ymax":tf.train.Feature(float_list=tf.train.FloatList(value=anno_[3]))
                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()
    
def decode_train_tfrecord():
    train_addr = os.path.join(out_path, "train.tfrecords")
    filename_queue = tf.train.string_input_producer([train_addr])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
                                       features={
                                           'img_raw':tf.FixedLenFeature([], tf.string),
                                           'xmin':tf.FixedLenFeature([], tf.float32),
                                           'xmax':tf.FixedLenFeature([], tf.float32)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)*(1./255)-0.5
    print("========get img========")
    test_anno = tf.cast(features['xmin'], tf.float32)
    print("========get anno========")
    return test_anno


if __name__ == '__main__':
    print("===========start here============\n")
    #gen_tfrecord()
    sess = tf.Session()
    t_anno_ = decode_train_tfrecord()
    t_anno = sess.run([t_anno_])
    img, t_anno = sess.run([img_, t_anno_])
    print("======================\n",
           #img,  '\n',
           #tf.shape(img),  '\n',
           t_anno_,  '\n',
           tf.shape(t_anno_),  '\n',
           #tf.shape(img[1]),
           '\n=====================\n')
    

