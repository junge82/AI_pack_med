# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
# python train.py config15/train_wt_ax.txt
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
Iterator = tf.data.Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet
from util.logger import *

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()


all_data_param = { 'T1':    {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'T1', 'filename_not_contains': ('T1c','SP_T1','SP_T1c')},
                   'T1c':   {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'T1c', 'filename_not_contains': 'SP_T1c'},
                   'T2':    {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'T2', 'filename_not_contains': 'SP_T2'},
                   'Flair': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'Flair', 'filename_not_contains': 'SP_Flair'},
                   'SP_T1': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'SP_T1', 'filename_not_contains': 'SP_T1c'},
                   'SP_T1c': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'SP_T1c'},
                   'SP_T2': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase//BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'SP_T2'},
                   'SP_Flair': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'SP_Flair'},
                   'label': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names/HGG'),
                         'filename_contains': 'Label'}
                 }

soft_data_param = { 
                   'SP_T1': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/HGG'),
                         'filename_contains': 'SP_T1', 'filename_not_contains': 'SP_T1c'},
                   'SP_T1c': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/HGG'),
                         'filename_contains': 'SP_T1c'},
                   'SP_T2': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/LGG','/media/hdd/Meddatabase//BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/HGG'),
                         'filename_contains': 'SP_T2'},
                   'SP_Flair': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/HGG'),
                         'filename_contains': 'SP_Flair'},
                   'label': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/sp_train_names/HGG'),
                         'filename_contains': 'Label'}
                 }


half_data_param =   {'T1':    {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'T1', 'filename_not_contains': ('T1c','SP_T1','SP_T1c')},
                    'T1c':   {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'T1c', 'filename_not_contains': 'SP_T1c'},
                    'T2':    {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'T2', 'filename_not_contains': 'SP_T2'},
                    'Flair': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'Flair', 'filename_not_contains': 'SP_Flair'},
                    'SP_T1': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'SP_T1', 'filename_not_contains': 'SP_T1c'},
                    'SP_T1c': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'SP_T1c'},
                    'SP_T2': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase//BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'SP_T2'},
                    'SP_Flair': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'SP_Flair'},
                    'label': {'path_to_search': ('/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/LGG','/media/hdd/Meddatabase/BRATS2015_VSD/Dataset/Brats15_sp40_training/train_names_few/HGG'),
                         'filename_contains': 'Label'}
                    }

def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    set_logger('training.log')
    tf.logging.info('training started') 

    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)

    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64,   shape = full_label_shape)

    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby    = tf.nn.softmax(predicty)

    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    print('size of predicty:',predicty)

    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()

    dataloader = DataLoader(config_data)
    dataloader.load_data()
    validdataloader = DataLoader(config_data)
    validdataloader.valid_load_data()

    best_loss = 0.4
    early_stopping_step = 6
    should_stop = False
    stopping_step = 0
    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        # print("tempx: ",np.array(tempx).shape)
        # print("tempw: ",np.array(tempw).shape)
        # print("tempy: ",np.array(tempy).shape)
        opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy})
        print("test_iteration for training: ",n%config_train['test_iteration'])
        if(n%config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
                #print("training dice: ",dice)
                batch_dice_list.append(dice)
 
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))
            str = "t: {}, n: {}, train mean dice loss: {}".format(t , n, batch_dice) 
            print(str)
            tf.logging.info(str)

            # validation
            val_dice_list = []
            for step in range(20):
                valid_pair = validdataloader.validation_get_subimage_batch()            
                vtempx = valid_pair['images']
                vtempw = valid_pair['weights']
                vtempy = valid_pair['labels']
                val_dice = loss.eval(feed_dict ={x:vtempx, w:vtempw, y:vtempy})
                #print("validation dice: ",val_dice)
                val_dice_list.append(val_dice)

            val_batch_dice = np.asarray(val_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            str = "t: {}, n: {}, validation mean dice loss: {}".format(t,n,val_batch_dice)
            print(str)
            tf.logging.info(str)

            # early stopping
            if(val_batch_dice < best_loss):
                best_loss = val_batch_dice
                best_sess = sess
                #print ("best_loss: ", best_loss)
                str = "best_loss: {}".format(best_loss)  
                tf.logging.info(str)
                stopping_step = 0
            else:
                stopping_step +=1                
                print("stopping_step: ",stopping_step)
                str = "stopping_step: {}".format(stopping_step)  
                tf.logging.info(str)
            if stopping_step >= early_stopping_step: 
                should_stop = True
                str = "Early stopping is trigger at step: {} loss:{}".format(n,best_loss)  
                print(str)  
                tf.logging.info(str)
                saver.save(best_sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(10000))
                sess.close()
                break      

        if((n+1)%config_train['snapshot_iteration']  == 0 ):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
            tf.logging.info('training ended')
    sess.close()

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)