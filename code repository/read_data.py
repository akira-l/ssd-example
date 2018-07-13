import numpy as np
import os
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet import image


class get_data(object):
    def __init__(self):
        self.img_path = "/home/liang/SSD/ssd/wtf"
        self.anno_path = "/home/liang/SSD/ssd/"
    
    
    def read_images(self, img_name):
        print('%s/%s.jpg' % (self.img_path, img_name))
        img = image.imread('%s/%s.jpg' % (self.img_path, img_name))
        return img
        
        
    def read_anno(self, anno_name):
        anno = np.load(self.anno_path+anno_name)
        '''
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        '''
        return anno
        
        


if __name__ == '__main__':
    num = 2
    fetch_data = get_data()
    print("1???")
    fetch_data.read_images(str(num))
    print("2???")
    anno = fetch_data.read_anno("label.npy")
    print("3???")
    anno_ = anno[num]
    print("====image====")
    print(fetch_data)
    print("====anno====")
    print(anno_,"\n    len    \n",len(anno_))

