# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import os
import random
import nibabel
import numpy as np
from scipy import ndimage
from util.data_process import *
from niftynet.io.image_reader import ImageReader
import matplotlib.pyplot as plt
import tensorflow as tf

class DataLoader():
    def __init__(self, config):
        """
        Initialize the calss instance
        inputs:
            config: a dictionary representing parameters
        """
        self.config    = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [config['data_root']]
        self.modality_postfix     = config.get('modality_postfix', ['flair', 't1', 't1c', 't2','sp_flair', 'sp_t1', 'sp_t1c', 'sp_t2'])
        #self.modality_postfix     = config.get('modality_postfix', ['sp_flair', 'sp_t1', 'sp_t1c', 'sp_t2'])
        self.intensity_normalize  = config.get('intensity_normalize', [True, True, True, True, True, True, True, True])
        #self.intensity_normalize  = config.get('intensity_normalize', [True, True, True, True])
        self.with_ground_truth    = config.get('with_ground_truth', False)
        self.label_convert_source = config.get('label_convert_source', None)
        self.label_convert_target = config.get('label_convert_target', None)
        self.label_postfix = config.get('label_postfix', 'label')
        self.file_postfix  = config.get('file_postfix', 'nii.gz')
        self.data_names    = config.get('data_names', None)
        self.validation_data_names    = config.get('validation_data_names', None)
        self.data_num      = config.get('data_num', None)
        self.vdata_num      = config.get('vdata_num', None)
        self.data_resize   = config.get('data_resize', None)
        self.with_flip     = config.get('with_flip', False)

        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))

    def __valid_get_patient_names(self):
        """
        get the list of patient names, if self.validation_data_names id not None, then load patient
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        if(self.validation_data_names is not None):
            tf.logging.info('predefined datanames')
            #print("datanames: ",self.validation_data_names)
            assert(os.path.isfile(self.validation_data_names))
            with open(self.validation_data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]
        # use all the patient names in data_root
        else:
            tf.logging.info('else readdir datanames')
            patient_names = os.listdir(self.data_root[0])
            patient_names = [name for name in patient_names if 'brats' in name.lower()]
        #print("patient name ", patient_names)
        return patient_names

    def __get_patient_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        if(self.data_names is not None):
            tf.logging.info('predefined datanames')
            print("datanames: ",self.data_names)
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]
        # use all the patient names in data_root
        else:
            tf.logging.info('else datanames')
            patient_names = os.listdir(self.data_root[0])
            patient_names = [name for name in patient_names if 'brats' in name.lower()]
        #print("patient name ", patient_names)
        return patient_names

    def __load_one_volume(self, patient_name, mod):
        patient_dir = os.path.join(self.data_root[0], patient_name)
        # for brats17
        if('nii' in self.file_postfix):
            #print("**17*")
            image_names = os.listdir(patient_dir)
            volume_name = None
            for image_name in image_names:
                if('SP' not in mod):
                    if(mod + '.' in image_name and 'SP' not in image_name):
                        volume_name = image_name
                        break
                else :
                    if(mod + '.' in image_name):
                        #print("im_name:",image_name)
                        volume_name = image_name
                        break
        # for brats15
        else:
            #print("**15*")
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + self.file_postfix
                    #print("volume_name: ", volume_name)
                    break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_dir, volume_name)
        volume = load_3d_volume_as_array(volume_name)
        return volume, volume_name

    def load_data_nifty(self,data_param):
          grouping_param = {'image': ('T1', 'T1c', 'T2', 'Flair','T1', 'SP_T1c', 'SP_T2', 'SP_Flair'), 'label':('label',) }
          #grouping_param = {'image': ('SP_T1', 'SP_T1c', 'SP_T2', 'SP_Flair'), 'label':('label',) }
    
          reader = ImageReader().initialise(data_param, grouping_param)
          self.patient_names = self.__get_patient_names()
          assert(len(self.patient_names)  > 0)
                #print("number of subjects: ", reader.num_subjects)
          ImageNames = []
          X = []
          W = []
          Y = []
          bbox  = []
          in_size = []

          data_num = reader.num_subjects
                #print("data_num: ",reader.num_subjects)
          for i in range(reader.num_subjects) :
                    volume_list = []
                    volume_name_list = []
                    idx, image_data, interp_order = reader(idx=i)
                    #print("image_data: ",len(image_data))
                    for mod_idx, modality_postfix in enumerate(self.modality_postfix):
                            #print("mod_idx: ",mod_idx)
                        volume = image_data['image'][:,:,:,:,mod_idx]
                        volume = volume[:,:,:,0]
                        volume = np.transpose(volume, (2,1,0))
                        volume_name = reader.get_subject(i)[modality_postfix]
                        #plt.imshow(volume[:,:,56])
                        #plt.show()
                        if(mod_idx == 0):
                            margin = 5
                            bbmin, bbmax = get_ND_bounding_box(volume, margin)
                            volume_size  = volume.shape
                            #print("volume size",volume_size)
                        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
                        #plt.imshow(volume[:,:,56])
                        #plt.show()
                        if(mod_idx ==0):
                            weight = np.asarray(volume > 0, np.float32)
                        if(self.intensity_normalize[mod_idx]):
                            volume = itensity_normalize_one_volume(volume)
                        #plt.imshow(volume[:,:,56])
                        #plt.show()
                        volume_list.append(volume)
                        volume_name_list.append(volume_name)
                        #print("volume: ",volume_name)
                        #print("volume: ",volume.shape)
                    ImageNames.append(volume_name_list)
                    X.append(volume_list)
                    #print("volumelist len: {}", len(volume_list))
                    #print("X shape: ",np.array(X).shape)
                    W.append(weight)
                    #print("weight shape :", np.array(weight).shape)
                    bbox.append([bbmin, bbmax])
                    in_size.append(volume_size)
                    if(self.with_ground_truth):
                        label = image_data['label'][:,:,:,:,0]
                        label = label[:,:,:,0]
                        label = np.transpose(label, (2,1,0))
                        #print("label", label.shape)
                        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
                        #print("label", label.shape)

                        # uniq = np.unique(label)
                        # print("uniq ",uniq)
                        # assert len(uniq) == 5
                        Y.append(label)
                    if((i+1)%50 == 0 or (i+1) == data_num):
                        print('Data load, {0:}% finished'.format((i+1)*100.0/data_num))
          self.image_names = ImageNames
          self.data   = X
          self.weight = W
          self.label  = Y
          self.bbox   = bbox
          self.in_size= in_size

    def valid_load_data(self):
        """
        load all the valid data
        """
        self.valid_patient_names = self.__valid_get_patient_names()
        assert(len(self.valid_patient_names)  > 0)
        print("valid_patient_names: ", len(self.valid_patient_names))
        ImageNames = []
        X = []
        W = []
        Y = []
        bbox  = []
        in_size = []
        data_num = self.vdata_num if (self.vdata_num is not None) else len(self.valid_patient_names)
        for i in range(data_num):
            volume_list = []
            volume_name_list = []
            for mod_idx, modality_postfix in enumerate(self.modality_postfix):
                volume, volume_name = self.__load_one_volume(self.valid_patient_names[i], modality_postfix)
                #print("volume: ",volume_name)
                #print("volume: ",volume.shape)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                if(mod_idx == 0):
                    margin = 5
                    bbmin, bbmax = get_ND_bounding_box(volume, margin)
                    volume_size  = volume.shape
                volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                if(mod_idx ==0):
                    weight = np.asarray(volume > 0, np.float32)
                if(self.intensity_normalize[mod_idx]):
                    volume = itensity_normalize_one_volume(volume)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                volume_list.append(volume)
                volume_name_list.append(volume_name)
                #print("volume: ",volume_name)
                #print("volume: ",volume.shape)
            ImageNames.append(volume_name_list)
            X.append(volume_list)
            #print("volumelist len: {}", len(volume_list))
            #print("X shape: ",np.array(X).shape)
            W.append(weight)
            #print("weight shape :", np.array(weight).shape)
            bbox.append([bbmin, bbmax])
            in_size.append(volume_size)
            if(self.with_ground_truth):

                label, labelname = self.__load_one_volume(self.valid_patient_names[i], self.label_postfix)
                label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)

                # uniq = np.unique(label)
                # print("uniq ",uniq)
                # assert len(uniq) == 5
                Y.append(label)
            if((i+1)%50 == 0 or (i+1) == data_num):
                print('Data load, {0:}% finished'.format((i+1)*100.0/data_num))
        self.vimage_names = ImageNames
        self.data   = X
        self.weight = W
        self.label  = Y
        self.bbox   = bbox
        self.in_size= in_size

    def load_data(self):
        """
        load all the training/testing data
        """
        self.patient_names = self.__get_patient_names()
        assert(len(self.patient_names)  > 0)
        print("patient_names: ", len(self.patient_names))
        ImageNames = []
        X = []
        W = []
        Y = []
        bbox  = []
        in_size = []
        data_num = self.data_num if (self.data_num is not None) else len(self.patient_names)
        for i in range(data_num):
            volume_list = []
            volume_name_list = []
            for mod_idx, modality_postfix in enumerate(self.modality_postfix):
                volume, volume_name = self.__load_one_volume(self.patient_names[i], modality_postfix)
                #print("volume: ",volume_name)
                #print("volume: ",volume.shape)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                if(mod_idx == 0):
                    margin = 5
                    bbmin, bbmax = get_ND_bounding_box(volume, margin)
                    volume_size  = volume.shape
                volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                if(mod_idx ==0):
                    weight = np.asarray(volume > 0, np.float32)
                if(self.intensity_normalize[mod_idx]):
                    volume = itensity_normalize_one_volume(volume)
                #plt.imshow(volume[:,:,56])
                #plt.show()
                volume_list.append(volume)
                volume_name_list.append(volume_name)
                #print("volume: ",volume_name)
                #print("volume: ",volume.shape)
            ImageNames.append(volume_name_list)
            X.append(volume_list)
            #print("volumelist len: {}", len(volume_list))
            #print("X shape: ",np.array(X).shape)
            W.append(weight)
            #print("weight shape :", np.array(weight).shape)
            bbox.append([bbmin, bbmax])
            in_size.append(volume_size)
            if(self.with_ground_truth):

                label, labelname = self.__load_one_volume(self.patient_names[i], self.label_postfix)
                label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)

                # uniq = np.unique(label)
                # print("uniq ",uniq)
                # assert len(uniq) == 5
                Y.append(label)
            if((i+1)%50 == 0 or (i+1) == data_num):
                print('Data load, {0:}% finished'.format((i+1)*100.0/data_num))
        self.image_names = ImageNames
        self.data   = X
        self.weight = W
        self.label  = Y
        self.bbox   = bbox
        self.in_size= in_size

    def get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        flag = False
        while(flag == False):
            batch = self.__get_one_batch()
            labels = batch['labels']
            if(labels.sum() > 0):
                flag = True
        return batch

    def validation_get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        flag = False
        while(flag == False):
            batch = self.__get_one_batch(True)
            labels = batch['labels']
            if(labels.sum() > 0):
                flag = True
        return batch

    def __get_one_batch(self, validation=False):
        """
        get a batch from training data
        """
        batch_size = self.config['batch_size']
        data_shape = self.config['data_shape']
        label_shape = self.config['label_shape']
        down_sample_rate = self.config.get('down_sample_rate', 1.0)
        data_slice_number = data_shape[0]
        label_slice_number = label_shape[0]
        batch_sample_model   = self.config.get('batch_sample_model', ('full', 'valid', 'valid'))
        batch_slice_direction= self.config.get('batch_slice_direction', 'axial') # axial, sagittal, coronal or random
        train_with_roi_patch = self.config.get('train_with_roi_patch', False)
        keep_roi_outside = self.config.get('keep_roi_outside', False)
        if(train_with_roi_patch):
            label_roi_mask    = self.config['label_roi_mask']
            roi_patch_margin  = self.config['roi_patch_margin']

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        weight_batch = []
        label_batch = []
        slice_direction = batch_slice_direction
        if(slice_direction == 'random'):
            directions = ['axial', 'sagittal', 'coronal']
            idx = random.randint(0,2)
            slice_direction = directions[idx]
            print("slice_direction: ", slice_direction)
        for i in range(batch_size):

            if(self.with_flip):
                flip = random.random() > 0.5
            else:
                flip = False              
            if validation: 
                self.patient_id = random.randint(0, len(self.data)-1)
                #print("len: ",len(self.data)-1)
                #print("patient_id ", self.patient_id)
                data_volumes = [x for x in self.data[self.patient_id]]

            else : 
                self.patient_id = random.randint(0, len(self.data)-1)
                #print("len: ",len(self.data)-1)
                #print("patient_id ", self.patient_id)
                data_volumes = [x for x in self.data[self.patient_id]]

            #print("data_volumes shape: ",np.array(data_volumes).shape)
            weight_volumes = [self.weight[self.patient_id]]
            #print("weight_volumes shape: ",np.array(weight_volumes).shape)
            boundingbox = None
            if(self.with_ground_truth):
                label_volumes = [self.label[self.patient_id]]
                if(train_with_roi_patch):
                    mask_volume = np.zeros_like(label_volumes[0])
                    for mask_label in label_roi_mask:
                        mask_volume = mask_volume + (label_volumes[0] == mask_label)
                    [d_idxes, h_idxes, w_idxes] = np.nonzero(mask_volume)
                    [D, H, W] = label_volumes[0].shape
                    mind = max(d_idxes.min() - roi_patch_margin, 0)
                    maxd = min(d_idxes.max() + roi_patch_margin, D)
                    minh = max(h_idxes.min() - roi_patch_margin, 0)
                    maxh = min(h_idxes.max() + roi_patch_margin, H)
                    minw = max(w_idxes.min() - roi_patch_margin, 0)
                    maxw = min(w_idxes.max() + roi_patch_margin, W)
                    if(keep_roi_outside):
                        boundingbox = [mind, maxd, minh, maxh, minw, maxw]
                    else:
                        for idx in range(len(data_volumes)):
                            data_volumes[idx] = data_volumes[idx][np.ix_(range(mind, maxd),
                                                                     range(minh, maxh),
                                                                     range(minw, maxw))]
                        for idx in range(len(weight_volumes)):
                            weight_volumes[idx] = weight_volumes[idx][np.ix_(range(mind, maxd),
                                                                     range(minh, maxh),
                                                                     range(minw, maxw))]
                        for idx in range(len(label_volumes)):
                            label_volumes[idx] = label_volumes[idx][np.ix_(range(mind, maxd),
                                                                     range(minh, maxh),
                                                                     range(minw, maxw))]

                if(self.label_convert_source and self.label_convert_target):
                    label_volumes[0] = convert_label(label_volumes[0], self.label_convert_source, self.label_convert_target)

            transposed_volumes = transpose_volumes(data_volumes, slice_direction)
            volume_shape = transposed_volumes[0].shape
            #print("data shape: 1 {} 2 {}", np.array(data_shape[1]).shape, np.array(data_shape[2]).shape )
            sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]
            center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, batch_sample_model, boundingbox)
            sub_data = []
            for moda in range(len(transposed_volumes)):
                sub_data_moda = extract_roi_from_volume(transposed_volumes[moda],center_point,sub_data_shape)
                if(flip):
                    sub_data_moda = np.flip(sub_data_moda, -1)
                if(down_sample_rate != 1.0):
                    sub_data_moda = ndimage.interpolation.zoom(sub_data_moda, 1.0/down_sample_rate, order = 1)
                sub_data.append(sub_data_moda)
            sub_data = np.asarray(sub_data)
            data_batch.append(sub_data)
            transposed_weight = transpose_volumes(weight_volumes, slice_direction)
            sub_weight = extract_roi_from_volume(transposed_weight[0],
                                                  center_point,
                                                  sub_label_shape,
                                                  fill = 'zero')

            if(flip):
                sub_weight = np.flip(sub_weight, -1)
            if(down_sample_rate != 1.0):
                    sub_weight = ndimage.interpolation.zoom(sub_weight, 1.0/down_sample_rate, order = 1)
            weight_batch.append([sub_weight])
            if(self.with_ground_truth):
                tranposed_label = transpose_volumes(label_volumes, slice_direction)
                sub_label = extract_roi_from_volume(tranposed_label[0],
                                                     center_point,
                                                     sub_label_shape,
                                                     fill = 'zero')
                if(flip):
                    sub_label = np.flip(sub_label, -1)
                if(down_sample_rate != 1.0):
                    sub_label = ndimage.interpolation.zoom(sub_label, 1.0/down_sample_rate, order = 0)
                label_batch.append([sub_label])

        data_batch = np.asarray(data_batch, np.float32)
        #print("data_batch ",data_batch.shape)
        weight_batch = np.asarray(weight_batch, np.float32)
        label_batch = np.asarray(label_batch, np.int64)
        batch = {}
        batch['images']  = np.transpose(data_batch,   [0, 2, 3, 4, 1])
        batch['weights'] = np.transpose(weight_batch, [0, 2, 3, 4, 1])
        batch['labels']  = np.transpose(label_batch,  [0, 2, 3, 4, 1])
        #print("batch len, keys, values:", len(batch), list(batch.keys()), np.array(batch['images'][0]).shape )
        return batch

    def get_total_image_number(self):
        """
        get the toal number of images
        """
        return len(self.data)

    def get_image_data_with_name(self, i):
        """
        Used for testing, get one image data and patient name
        """
        return [self.data[i], self.weight[i], patient_name[i], self.image_names[i], self.bbox[i], self.in_size[i]]
    
    def get_image_data_with_niftyname(self, i):
        """
        Used for testing, get one image data and patient name
        """
        #print("imagenames: ",self.image_names[i][0])
        volume_name = os.path.basename(os.path.normpath(self.image_names[i][4]))
        str = "volume_name: {}".format(volume_name)  
        tf.logging.info(str) 
        #print("volume name: ",volume_name)
        if '_SP_Flair' in volume_name :
            ground_truth = volume_name.replace('_SP_Flair','_Label')            
            print(ground_truth)
            str = "ground_truth: {}".format(ground_truth)  
            tf.logging.info(str) 
        return [self.data[i], self.weight[i], ground_truth, self.image_names[i], self.bbox[i], self.in_size[i]]