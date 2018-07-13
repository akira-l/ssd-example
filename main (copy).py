import os
import gc
import xml.etree.ElementTree as etxml
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import ssd_resnet2
import time
import PIL.Image
import skimage.io
import skimage.transform


stage_ = 'train'
data_path = "/home/liang/wider face/WIDER_"+stage_+"/images"

def testing():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd_resnet2.SSD_resnet(sess,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            saver.restore(sess, './session_params/session.ckpt')
            image, actual = get_data(1)
            pred_class, pred_class_val, pred_location = ssd_model.run(image,None)
            
            for index, act in zip(range(len(image)), actual):
                for a in act :
                    print('img-'+str(index)+' actual:' + str(a))
                print('pred_class:' + str(pred_class[index]))
                print('pred_class_val:' + str(pred_class_val[index]))
                print('pred_location:' + str(pred_location[index]))   
                                   
        else:
            print('No Data Exists!')
        sess.close()


def training():
    batch_size = 5
    running_count = 0
    
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd_resnet2.SSD_resnet(sess,True)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            print('\nStart Restore')
            saver.restore(sess, './session_params/session.ckpt')
            print('\nEnd Restore')
        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.
        while((min_loss_location + min_loss_class) > 0.001 and running_count < 100000):
            running_count += 1
            
            train_data, actual_data = get_data(batch_size)
            if len(train_data) > 0:
                loss_all,loss_class,loss_location,pred_class,pred_location = ssd_model.run(train_data, actual_data)
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                if min_loss_location > l:
                    min_loss_location = l
                if min_loss_class > c:
                    min_loss_class = c
               
                print('Running:' + str(running_count) + '|Loss All:'+str(min_loss_location + min_loss_class)+'|'+ str(loss_all) + '|Location:'+ str(np.sum(loss_location)) + '|Class:'+ str(np.sum(loss_class)) + '|pred_class:'+ str(np.sum(pred_class))+'|'+str(np.amax(pred_class))+'|'+ str(np.min(pred_class)) + '|pred_location:'+ str(np.sum(pred_location))+'|'+str(np.amax(pred_location))+'|'+ str(np.min(pred_location)) + '')
                
                if running_count % 100 == 0:
                    saver.save(sess, './session_params/session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break
            
        saver.save(sess, './session_params/session.ckpt')
        sess.close()
        gc.collect()
            
    print('End Training')
    

def get_data(batch_size):
    batch_list = random.sample(range(len(name_list)), batch_size)
    train_data = []
    anno_data = []
    for img_num in batch_list:
        #img = PIL.Image.open(os.path.join(data_path, name_list[img_num]))
        img = skimage.io.imread(os.path.join(data_path, name_list[img_num]))
        height = img.shape[0]
        width = img.shape[1]
        #width, height = img.size
        img = skimage.transform.resize(img, (300, 300))
        #img.resize((300,300),PIL.Image.ANTIALIAS)
        #img = img-whitened_RGB_mean
        
        img = img/255 - 0.5
        train_data.append(np.array(img))
        
        anno_batch = anno_list[img_num]
        anno_sub = []
        for anno_num in range(len(anno_batch)):
            [x_min, y_min, a_w, a_h] = anno_batch[anno_num]
            cen_x = (x_min+a_w/2)/width
            cen_y = (y_min+a_h/2)/height
            a_w = a_w / width
            a_h = a_h / height
            
            anno_sub.append([cen_x, cen_y, a_w, a_h, 1])#label =1
        anno_data.append(anno_sub)
    #anno_data = float(anno_data)*(1./255)-0.5
    return train_data, anno_data
    
if __name__ == '__main__':
    file_name_addr = "/home/liang/wider face/wider_face_split/wider_face_"+ stage_ +"_bbx_gt.txt"
    name_file = open(file_name_addr)
    
    name_list = []
    anno_list = []
    
    line_cache = ' '
    while line_cache:
        line_cache = name_file.readline()
        if line_cache[-5:] == '.jpg\n':
            name_list.append(line_cache[:-1])
            line_cache = name_file.readline()
            anno_sub_list = []
            for counter_in_line in range(int(line_cache)):
                line_cache_pos = name_file.readline()
                anno_sub_list.append([int(x) for x in line_cache_pos.split()[:4]])
            anno_list.append(anno_sub_list)
    
    print('\nStart Running')
    #testing()
    training()
    print('\nEnd Running')
