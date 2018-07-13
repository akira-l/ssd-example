import numpy as np
import tensorflow as tf
import os
from resnet import *

from tensorflow.python.ops import array_ops
from tensorflow.python.training.moving_averages import assign_moving_average

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
        c2_1 = self.convolution(p2, [1, 1, 512, 256], self.conv_strides_1, 'features_c2_1')
        c2_2 = tf.image.resize_bilinear(p1, [19, 19], name='sub_net2-2')
        c2_2 = self.convolution(c2_2, [1, 1, 1024, 256], self.conv_strides_1, 'features_c2_2')
        self.c2 = tf.add(c2_1, c2_2, name='c2_layer')
        
        c3_1 = self.convolution(p3, [1,1,256,256], self.conv_strides_1, 'features_c3_1')
        c3_2 = tf.image.resize_bilinear(p2, [38,38], name='sub_net3-2')
        c3_2 = self.convolution(c3_2, [1,1,512, 256], self.conv_strides_1, 'features_c3_2')
        self.c3 = tf.add(c3_1, c3_2, name='c3_layer')
        
        self.features_1 = self.convolution(self.c1, [3, 3, 1024, self.default_box_size[0]*(self.classes_size + 4)], self.conv_strides_1, 'features_1')
        print('##   features_1 shape: ' + str(self.features_1.get_shape().as_list()))
        self.features_2 = self.convolution(self.c2, [3, 3, 256, self.default_box_size[1]*(self.classes_size + 4)], self.conv_strides_1, 'features_2')
        print('##   features_2 shape: ' + str(self.features_2.get_shape().as_list()))
        self.features_3 = self.convolution(self.c3, [3, 3, 256, self.default_box_size[2]*(self.classes_size + 4)], self.conv_strides_1, 'features_3')
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

        # loss
        
        self.loss_location = tf.reduce_sum(self.smooth_L1(tf.subtract(self.groundtruth_location , self.feature_location)), reduction_indices=2)
        
        
        
        ####this is huber loss 
        ####same to the smooth l1 loss
        #delta1 = tf.constant(1)
        #self.loss_location = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((tf.subtract(self.groundtruth_location, self.feature_location))/delta1)) - 1.)
        
        #self.loss_class = tf.div(tf.reduce_sum(tf.multiply(self.softmax_cross_entropy , self.groundtruth_count), reduction_indices=1) , tf.reduce_sum(self.groundtruth_count, reduction_indices = 1))
        
        self.loss_class = self.focal_loss(self.feature_class, self.groundtruth_class)
        
        self.loss_all = tf.reduce_sum(tf.add(self.loss_class , self.loss_location))
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

            print("==========shape of input_image", len(input_images))

            f_class, f_location = self.sess.run([self.feature_class, self.feature_location], feed_dict={self.images : input_images })
            
            print("===============================")

            with tf.control_dependencies([self.feature_class, self.feature_location]):
            
                f_class = self.check_numerics(f_class,'predf_class')
                f_location = self.check_numerics(f_location,'predf_location')
                
                
                gt_class,gt_location,gt_data = self.generate_groundtruth_data(actual_data, f_class) 
                
                self.sess.run(self.train, feed_dict={
                    self.images : input_images, 
                    self.groundtruth_class : gt_class,
                    self.groundtruth_location : gt_location,
                    self.groundtruth: gt_data
                })
                with tf.control_dependencies([self.train]):
                    loss_all,loss_location,loss_class = self.sess.run([self.loss_all,self.loss_location,self.loss_class], feed_dict={
                        self.images : input_images,
                        self.groundtruth_class : gt_class,
                        self.groundtruth_location : gt_location,
                        self.groundtruth: gt_data
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
    
    
    def focal_loss(self,prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
        prediction_tensor = tf.convert_to_tensor(tf.cast(prediction_tensor, tf.float32), tf.float32)
        target_tensor = tf.convert_to_tensor(tf.cast(target_tensor, tf.float32), tf.float32)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.feature_class, labels=self.groundtruth_class)
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
    
    
    
    # conv
    def convolution(self, input, shape, strides, name):
        with tf.variable_scope(name):
            weight = tf.get_variable(initializer=tf.truncated_normal(shape, 0, 1), dtype=tf.float32, name=name+'_weight')
            bias = tf.get_variable(initializer=tf.truncated_normal(shape[-1:], 0, 1), dtype=tf.float32, name=name+'_bias')
            result = tf.nn.conv2d(input, weight, strides, padding='SAME', name=name+'_conv')
            result = tf.nn.bias_add(result, bias)
            result = self.batch_normalization(result, name=name+'_bn')
            result = tf.nn.relu(result, name=name+'_relu')
            return result

    # fully connect
    def fc(self, input, out_shape, name):
        with tf.variable_scope(name+'_fc'):
            in_shape = 1
            for d in input.get_shape().as_list()[1:]:
                in_shape *= d
            weight = tf.get_variable(initializer=tf.truncated_normal([in_shape, out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_weight')
            bias = tf.get_variable(initializer=tf.truncated_normal([out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_bias')
            result = tf.reshape(input, [-1, in_shape])
            result = tf.nn.xw_plus_b(result, weight, bias, name=name+'_fc_do')
            return result

    # Batch Normalization
    def batch_normalization(self, input, name):
        with tf.variable_scope(name):
            bn_input_shape = input.get_shape() 
            moving_mean = tf.get_variable(name+'_mean', bn_input_shape[-1:] , initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable(name+'_variance', bn_input_shape[-1:] , initializer=tf.ones_initializer, trainable=False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(input, list(range(len(bn_input_shape) - 1)), name=name+'_moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, self.conv_bn_decay),assign_moving_average(moving_variance, variance, self.conv_bn_decay)]):
                    return tf.identity(mean), tf.identity(variance)
            #mean, variance = tf.cond(tf.cast(self.isTraining, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            mean, variance = tf.cond(tf.cast(True, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            beta = tf.get_variable(name+'_beta', bn_input_shape[-1:] , initializer=tf.zeros_initializer)
            gamma = tf.get_variable(name+'_gamma', bn_input_shape[-1:] , initializer=tf.ones_initializer)
            return tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.conv_bn_epsilon, name+'_bn_opt')
    
    # smooth_L1 
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))
        
    # 
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
