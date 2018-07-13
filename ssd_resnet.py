import numpy as np
import tensorflow as tf
from resnet import *
from tools import *
from loss import *

class SSD_resnet:
    def __init__(self, tf_sess, isTraining):
        # tensorflow session
        self.sess = tf_sess
        self.isTraining = isTraining
        self.img_size = [300, 300]
        self.classes_size = 2
        self.background_classes_val = 0
        self.default_box_size = [4, 6, 6, 6, 4, 4]
        self.box_aspect_ratio = [
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0]
        ]
        
        self.min_box_scale = 0.05
        self.max_box_scale = 0.9
        self.default_box_scale = np.linspace(self.min_box_scale, self.max_box_scale, num = np.amax(self.default_box_size))
        print('##   default_box_scale:'+str(self.default_box_scale))
        self.conv_strides_1 = [1, 1, 1, 1]
        self.conv_strides_2 = [1, 2, 2, 1]
        self.conv_strides_3 = [1, 3, 3, 1]
        self.pool_size = [1, 2, 2, 1]
        self.pool_strides = [1, 2, 2, 1]
        self.conv_bn_decay = 0.99999
        self.conv_bn_epsilon = 0.00001
        self.jaccard_value = 0.6
        
        self.generate_graph()
        
        
        
        
    def generate_graph(self):
        self.images = tf.placeholder(shape=[None, self.img_size[0], self.img_size[1], 3], dtype=tf.float32, name='input_image')
        
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, sub_net, end_points = resnet_v2_50(self.images,1001)
            saver1 = tf.train.Saver(tf.global_variables())  
            checkpoint_path = './model/resnet_v2_50.ckpt'  
            saver1.restore(self.sess, checkpoint_path) 
        
        # sub_net 0 : 38 38 256
        # sub_net 1 : 19 19 512
        # sub_net 2 : 10 10 1024
        # sub_net 3 : 10 10 2048
        
        p1 = sub_net[2]
        p2 = sub_net[1]
        p3 = sub_net[0]
        
        self.c1 = p1
        c2_1 = convolution(p2, [1, 1, 512, 256], self.conv_strides_1, 'features_c2_1')
        c2_2 = tf.image.resize_bilinear(p1, [19, 19], name='sub_net2-2')
        c2_2 = convolution(c2_2, [1, 1, 1024, 256], self.conv_strides_1, 'features_c2_2')
        self.c2 = tf.add(c2_1, c2_2, name='c2_layer')
        
        c3_1 = convolution(p3, [1,1,256,256], self.conv_strides_1, 'features_c3_1')
        c3_2 = tf.image.resize_bilinear(p2, [38,38], name='sub_net3-2')
        c3_2 = convolution(c3_2, [1,1,512, 256], self.conv_strides_1, 'features_c3_2')
        self.c3 = tf.add(c3_1, c3_2, name='c3_layer')
        
        
        
        #attention module
        att_c1_1 = convolution(self.c1, [3, 3, 1024, 256], self.conv_strides_1, 'attention_c1_1')
        att_c1_2 = convolution(att_c1_1, [3, 3, 256, 256], self.conv_strides_1, 'attention_c1_2')
        att_c1_3 = convolution(att_c1_2, [3, 3, 256, 256], self.conv_strides_1, 'attention_c1_3')
        att_c1_4 = convolution(att_c1_3, [3, 3, 256, 256], self.conv_strides_1, 'attention_c1_4')
        self.attention_c1 = convolution(att_c1_4, [3, 3, 256, 1], self.conv_strides_1, 'attention_c1')
        
        att_c2_1 = convolution(self.c2, [3, 3, 256, 256], self.conv_strides_1, 'attention_c2_1')
        att_c2_2 = convolution(att_c2_1, [3, 3, 256, 256], self.conv_strides_1, 'attention_c2_2')
        att_c2_3 = convolution(att_c2_2, [3, 3, 256, 256], self.conv_strides_1, 'attention_c2_3')
        att_c2_4 = convolution(att_c2_3, [3, 3, 256, 256], self.conv_strides_1, 'attention_c2_4')
        self.attention_c2 = convolution(att_c2_4, [3, 3, 256, 1], self.conv_strides_1, 'attention_c2')
        
        att_c3_1 = convolution(self.c3, [3, 3, 256, 256], self.conv_strides_1, 'attention_c3_1')
        att_c3_2 = convolution(att_c3_1, [3, 3, 256, 256], self.conv_strides_1, 'attention_c3_2')
        att_c3_3 = convolution(att_c3_2, [3, 3, 256, 256], self.conv_strides_1, 'attention_c3_3')
        att_c3_4 = convolution(att_c3_3, [3, 3, 256, 256], self.conv_strides_1, 'attention_c3_4')
        self.attention_c3 = convolution(att_c3_4, [3, 3, 256, 1], self.conv_strides_1, 'attention_c3')
        
        self.c1 = tf.multiply(self.c1, tf.exp(self.attention_c1))
        self.c2 = tf.multiply(self.c2, tf.exp(self.attention_c2))
        self.c3 = tf.multiply(self.c3, tf.exp(self.attention_c3))
        
        self.attention_c1 = tf.reshape(self.attention_c1, [-1, 10, 10])
        self.attention_c2 = tf.reshape(self.attention_c2, [-1, 19, 19])
        self.attention_c3 = tf.reshape(self.attention_c3, [-1, 38, 38])
        
        self.features_1 = convolution(self.c1, [3, 3, 1024, self.default_box_size[0]*(self.classes_size + 4)], self.conv_strides_1, 'features_1')
        print('##   features_1 shape: ' + str(self.features_1.get_shape().as_list()))
        self.features_2 = convolution(self.c2, [3, 3, 256, self.default_box_size[1]*(self.classes_size + 4)], self.conv_strides_1, 'features_2')
        print('##   features_2 shape: ' + str(self.features_2.get_shape().as_list()))
        self.features_3 = convolution(self.c3, [3, 3, 256, self.default_box_size[2]*(self.classes_size + 4)], self.conv_strides_1, 'features_3')
        print('##   features_3 shape: ' + str(self.features_3.get_shape().as_list()))
        
        
        self.feature_maps = [self.features_1, self.features_2, self.features_3]
        
        self.feature_maps_shape = [m.get_shape().as_list() for m in self.feature_maps]
        
        self.tmp_all_feature = []
        for i, fmap in zip(range(len(self.feature_maps)), self.feature_maps):
            width = self.feature_maps_shape[i][1]
            height = self.feature_maps_shape[i][2]
            self.tmp_all_feature.append(tf.reshape(fmap, [-1, (width * height * self.default_box_size[i]) , (self.classes_size + 4)]))
        
        self.tmp_all_feature = tf.concat(self.tmp_all_feature, axis=1)
        
        self.feature_class = self.tmp_all_feature[:,:,:self.classes_size]
        
        self.feature_location = self.tmp_all_feature[:,:,self.classes_size:]
        
        self.all_default_boxs = self.generate_all_default_boxs()
        self.all_default_boxs_len = len(self.all_default_boxs)
        # 
        self.groundtruth_class = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.int32,name='groundtruth_class')
        self.groundtruth_location = tf.placeholder(shape=[None,self.all_default_boxs_len,4], dtype=tf.float32,name='groundtruth_location')
        self.groundtruth = tf.placeholder(shape=[None,self.all_default_boxs_len, 2], dtype=tf.float32,name='groundtruth')
        self.groundtruth_attention1 = tf.placeholder(shape=[None, 10, 10], dtype=tf.float32,name='groundtruth_attention1')
        self.groundtruth_attention2 = tf.placeholder(shape=[None, 19, 19], dtype=tf.float32,name='groundtruth_attention2')
        self.groundtruth_attention3 = tf.placeholder(shape=[None, 38, 38], dtype=tf.float32,name='groundtruth_attention3')
        
        
        
        
        
        
        # loss
        self.attention_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.attention_c1, labels = self.groundtruth_attention1))
        self.attention_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.attention_c2, labels = self.groundtruth_attention2))
        self.attention_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.attention_c3, labels = self.groundtruth_attention3))
        
        self.loss_location = tf.reduce_sum(smooth_L1(tf.subtract(self.groundtruth_location , self.feature_location)), reduction_indices=2)
        
        ####this is huber loss 
        ####same to the smooth l1 loss
        #delta1 = tf.constant(1)
        #self.loss_location = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((tf.subtract(self.groundtruth_location, self.feature_location))/delta1)) - 1.)
        
        #self.loss_class = tf.div(tf.reduce_sum(tf.multiply(self.softmax_cross_entropy , self.groundtruth_count), reduction_indices=1) , tf.reduce_sum(self.groundtruth_count, reduction_indices = 1))
        
        self.loss_class = focal_loss(self.feature_class, self.groundtruth_class)
        
        self.loss_all = tf.reduce_sum(tf.add(tf.add(self.loss_class , self.loss_location),self.attention_loss))
        self.optimizer = tf.train.AdamOptimizer(0.001)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss_all)


    # 
    # input_images  [None,width,hight,channel]
    # actual_data [None,[None,center_X,center_Y,width,hight,classes]] , classes 0,classes_size)
    def run(self, input_images, actual_data):
        if self.isTraining :
            if actual_data is None :
                raise Exception('actual_data not exist')
            if len(input_images) != len(actual_data):
                raise Exception('input_images  actual_data not corresponding')

            f_class, f_location = self.sess.run([self.feature_class, self.feature_location], feed_dict={self.images : input_images })
            
            print("===============================")
            self.actual_data = actual_data

            with tf.control_dependencies([self.feature_class, self.feature_location]):
            
                f_class = self.check_numerics(f_class,'predf_class')
                f_location = self.check_numerics(f_location,'predf_location')
                
                gt_class,gt_location,gt_data = self.generate_groundtruth_data(actual_data, f_class) 
                
                
                attention1_gt = self.generate_attention(self.actual_data, [10,10])
                attention2_gt = self.generate_attention(self.actual_data, [19,19])
                attention3_gt = self.generate_attention(self.actual_data, [38,38])
                
                
                self.sess.run(self.train, feed_dict={
                    self.images : input_images, 
                    self.groundtruth_class : gt_class,
                    self.groundtruth_location : gt_location,
                    self.groundtruth: gt_data,
                    self.groundtruth_attention1: attention1_gt,
                    self.groundtruth_attention2: attention2_gt,
                    self.groundtruth_attention3: attention3_gt
                })
                with tf.control_dependencies([self.train]):
                    loss_all,loss_location,loss_class = self.sess.run([self.loss_all,self.loss_location,self.loss_class], feed_dict={
                        self.images : input_images,
                        self.groundtruth_class : gt_class,
                        self.groundtruth_location : gt_location,
                        self.groundtruth: gt_data,
                        self.groundtruth_attention1: attention1_gt,
                        self.groundtruth_attention2: attention2_gt,
                        self.groundtruth_attention3: attention3_gt
                    })
                    #
                    loss_all = self.check_numerics(loss_all,'loss_all') 
                    return loss_all, loss_class, loss_location, f_class, f_location
                    
        else :
            # softmax
            feature_class_softmax = tf.nn.softmax(logits=self.feature_class, dim=-1)
            # filter background
            background_filter = np.ones(self.classes_size, dtype=np.float32)
            background_filter[self.background_classes_val] = 0 
            background_filter = tf.constant(background_filter)  
            feature_class_softmax=tf.multiply(feature_class_softmax, background_filter)
            # max box
            feature_class_softmax = tf.reduce_max(feature_class_softmax,2)
            # 
            box_top_set = tf.nn.top_k(feature_class_softmax, int(self.all_default_boxs_len/20))
            box_top_index = box_top_set.indices
            box_top_value = box_top_set.values
            f_class, f_location, f_class_softmax, box_top_index, box_top_value = self.sess.run(
                [self.feature_class, self.feature_location, feature_class_softmax, box_top_index, box_top_value], 
                feed_dict={self.images : input_images }
            )
            top_shape = np.shape(box_top_index)
            pred_class = []
            pred_class_val = []
            pred_location = []
            for i in range(top_shape[0]) :
                item_img_class = []
                item_img_class_val = []
                item_img_location = []
                for j in range(top_shape[1]) : 
                    p_class_val = f_class_softmax[i][box_top_index[i][j]]
                    if p_class_val < 0.5:
                        continue
                    p_class = np.argmax(f_class[i][box_top_index[i][j]])
                    if p_class==self.background_classes_val:
                        continue
                    p_location = f_location[i][box_top_index[i][j]]
                    if p_location[0]<0 or p_location[1]<0 or p_location[2]<0 or p_location[3]<0 or p_location[2]==0 or p_location[3]==0 :
                        continue
                    is_box_filter = False
                    for f_index in range(len(item_img_class)) :
                        if self.jaccard(p_location,item_img_location[f_index]) > 0.3 and p_class == item_img_class[f_index] :
                            is_box_filter = True
                            break
                    if is_box_filter == False :
                        item_img_class.append(p_class)
                        item_img_class_val.append(p_class_val)
                        item_img_location.append(p_location)        
                pred_class.append(item_img_class)
                pred_class_val.append(item_img_class_val)
                pred_location.append(item_img_location)
            return pred_class, pred_class_val, pred_location
    
    
    
    def generate_all_default_boxs(self):
        
        all_default_boxes = []
        for index, map_shape in zip(range(len(self.feature_maps_shape)), self.feature_maps_shape):
            width = int(map_shape[1])
            height = int(map_shape[2])
            cell_scale = self.default_box_scale[index]
            for x in range(width):
                for y in range(height):
                    for ratio in self.box_aspect_ratio[index]:
                        center_x = (x / float(width)) + (0.5/ float(width))
                        center_y = (y / float(height)) + (0.5 / float(height))
                        box_width = np.sqrt(cell_scale * ratio)
                        box_height = np.sqrt(cell_scale / ratio)
                        all_default_boxes.append([center_x, center_y, box_width, box_height])
        all_default_boxes = np.array(all_default_boxes)
        
        all_default_boxes = self.check_numerics(all_default_boxes,'all_default_boxes') 
        return all_default_boxes
        
    def generate_attention(self, input_actual_data, feature_size):
        data_len = len(input_actual_data)
        attention_gt = np.zeros((data_len, feature_size[0], feature_size[1]))
        
        for i in range(data_len):
            for attention_ in input_actual_data[i]:
                cen_x = attention_[0]
                cen_y = attention_[1]
                a_w = attention_[2]
                a_h = attention_[3]
                
                if cen_x - a_w/2>0:
                    top_x = cen_x - a_w/2
                else:
                    top_x = 0
                if cen_y - a_h/2>0:
                    top_y = cen_y - a_h/2
                else:
                    top_y = 0
                    
                if cen_x + a_w/2>1:
                    end_x = cen_x + a_w/2
                else:
                    end_x = 1
                if cen_y + a_h/2>1:
                    end_y = cen_y + a_h/2
                else:
                    end_y = 1
                
                for coord_x in range(int(feature_size[0]*top_x),int(feature_size[0]*end_x)):
                    for coord_y in range(int(feature_size[1]*top_y), int(feature_size[1]*end_y)):
                        attention_gt[i, coord_x, coord_y] = 1
        return attention_gt
        
        
        
    def generate_groundtruth_data(self,input_actual_data, f_class):
        input_actual_data_len = len(input_actual_data)
        gt_class = np.zeros((input_actual_data_len, self.all_default_boxs_len)) 
        gt_location = np.zeros((input_actual_data_len, self.all_default_boxs_len, 4))
        gt_data = np.zeros((input_actual_data_len, self.all_default_boxs_len, 2))
        background_jacc = max(0, (self.jaccard_value-0.2))
        
        for img_index in range(input_actual_data_len):
            for pre_actual in input_actual_data[img_index]:
                gt_class_val = pre_actual[-1:][0]
                gt_box_val = pre_actual[:-1]
                for boxe_index in range(self.all_default_boxs_len):
                    jacc = self.jaccard(gt_box_val, self.all_default_boxs[boxe_index])
                    if jacc > self.jaccard_value or jacc == self.jaccard_value:
                        gt_class[img_index][boxe_index] = gt_class_val
                        gt_location[img_index][boxe_index] = gt_box_val
                        gt_data[img_index][0] = 1
                        gt_data[img_index][1] = 0
                    else:
                        gt_class[img_index][boxe_index] = gt_class_val
                        gt_location[img_index][boxe_index] = gt_box_val
                        gt_data[img_index][0] = 0
                        gt_data[img_index][1] = 1
        return gt_class, gt_location, gt_data
        
    # jaccard
    # IOU rect1 rect2 center_x,center_y,width,height]           
    def jaccard(self, rect1, rect2):
        x_overlap = max(0, (min(rect1[0]+(rect1[2]/2), rect2[0]+(rect2[2]/2)) - max(rect1[0]-(rect1[2]/2), rect2[0]-(rect2[2]/2))))
        y_overlap = max(0, (min(rect1[1]+(rect1[3]/2), rect2[1]+(rect2[3]/2)) - max(rect1[1]-(rect1[3]/2), rect2[1]-(rect2[3]/2))))
        intersection = x_overlap * y_overlap
        #
        rect1_width_sub = 0
        rect1_height_sub = 0
        rect2_width_sub = 0
        rect2_height_sub = 0
        if (rect1[0]-rect1[2]/2) < 0 : rect1_width_sub += 0-(rect1[0]-rect1[2]/2)
        if (rect1[0]+rect1[2]/2) > 1 : rect1_width_sub += (rect1[0]+rect1[2]/2)-1
        if (rect1[1]-rect1[3]/2) < 0 : rect1_height_sub += 0-(rect1[1]-rect1[3]/2)
        if (rect1[1]+rect1[3]/2) > 1 : rect1_height_sub += (rect1[1]+rect1[3]/2)-1
        if (rect2[0]-rect2[2]/2) < 0 : rect2_width_sub += 0-(rect2[0]-rect2[2]/2)
        if (rect2[0]+rect2[2]/2) > 1 : rect2_width_sub += (rect2[0]+rect2[2]/2)-1
        if (rect2[1]-rect2[3]/2) < 0 : rect2_height_sub += 0-(rect2[1]-rect2[3]/2)
        if (rect2[1]+rect2[3]/2) > 1 : rect2_height_sub += (rect2[1]+rect2[3]/2)-1
        area_box_a = (rect1[2]-rect1_width_sub) * (rect1[3]-rect1_height_sub)
        area_box_b = (rect2[2]-rect2_width_sub) * (rect2[3]-rect2_height_sub)
        union = area_box_a + area_box_b - intersection
        if intersection > 0 and union > 0 : 
            return intersection / union 
        else : 
            return 0
            
    # 
    def check_numerics(self, input_dataset, message):
        if str(input_dataset).find('Tensor') == 0 :
            input_dataset = tf.check_numerics(input_dataset, message)
        else :
            dataset = np.array(input_dataset)
            nan_count = np.count_nonzero(dataset != dataset) 
            inf_count = len(dataset[dataset == float("inf")])
            n_inf_count = len(dataset[dataset == float("-inf")])
            if nan_count>0 or inf_count>0 or n_inf_count>0:
                data_error = '['+ message +'error[nan:'+str(nan_count)+'|inf:'+str(inf_count)+'|-inf:'+str(n_inf_count)+']'
                raise Exception(data_error) 
        return  input_dataset
