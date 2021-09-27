import os
import sys
sys.path.insert(0, "/home/junge82/boruvka_superpixel/pybuild")

import boruvka_superpixel

import numpy as np
from scipy import ndimage
from data_process import *
from parse_config import parse_config

from multiprocessing import Pool
import skimage
import skimage.feature
import matplotlib.pyplot as plt

#python3 util/data_superpixel.py config15/train_wt_ax.txt
#python3 util/evaluation.py
#python3 test.py config15/test_wt_class.txt

def load_one_volume(patient_name, mod, data_root, file_postfix):
        patient_dir = os.path.join(data_root, patient_name)
        
        # for brats17
        if('nii' in file_postfix):
            image_names = os.listdir(patient_dir)
            #print("load_one_vol: ",image_names)
            volume_name = None
            for image_name in image_names:
                #print("mod", mod)
                if('SP' not in mod):
                    if(mod + '.' in image_name and 'SP' not in image_name):
                        #print("im_name:",image_name)
                        volume_name = image_name
                        break
                else :
                    if(mod + '.' in image_name):
                        #print("im_name:",image_name)
                        volume_name = image_name
                        break      
        # for brats15
        else:
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + file_postfix
                    break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_dir, volume_name)
        volume = load_3d_volume_as_array(volume_name)
        return volume, volume_name

def save_superpixel(patient_name, mod, sp_image, data_root, file_postfix):

    patient_dir = os.path.join(data_root, patient_name)
    patient_sp_dir = os.path.join(data_root, 'superpixel', patient_name)
    if not os.path.exists(patient_sp_dir):
        os.makedirs(patient_sp_dir)

    # for nifty
    if('nii' in file_postfix):
        image_names = os.listdir(patient_dir)
        volume_name = None
        for image_name in image_names:
            if(mod + '.' in image_name):                             
                volume_name = image_name.replace(mod,"SP")
                print("SP: ",volume_name)
                break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_sp_dir, volume_name)
        save_array_as_nifty_volume(sp_image,volume_name)
    # for mha
    else:
        img_file_dirs = os.listdir(patient_dir)
        #print("patient_dir")
        volume_name  = None
        for img_file_dir in img_file_dirs:
            if(mod+'.' in img_file_dir):
                volume_name = img_file_dir + '/' + img_file_dir + '.' + file_postfix
                if not os.path.exists(os.path.join(patient_sp_dir, img_file_dir)):
                    os.makedirs(os.path.join(patient_sp_dir, img_file_dir))
                break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_sp_dir, volume_name)
        save_array_volume_as_mha(volume_name, sp_image)
    

def save_separate(patient_name, volume_name, mod, sp_image, data_root, file_postfix):
    
        volume_name = os.path.basename(os.path.normpath(volume_name))
        #volume_name1 = '1' + volume_name 
        #volume_name2 = '2' + volume_name
        patient_dir = os.path.join(data_root, patient_name)    
        patient_HGG_dir = os.path.join(data_root, 'sp_train_names/HGG')
        patient_LGG_dir = os.path.join(data_root, 'sp_train_names/LGG')
        #print("sp_dir ",patient_HGG_dir)
        if not os.path.exists(patient_HGG_dir):
            os.makedirs(patient_HGG_dir)
        #print("sp_dir ",patient_LGG_dir)
        if not os.path.exists(patient_LGG_dir):
            os.makedirs(patient_LGG_dir)
        if 'HGG' in volume_name :
            patient_sp_dir = os.path.join(data_root, patient_HGG_dir, volume_name)
        if 'LGG' in volume_name :
            patient_sp_dir = os.path.join(data_root, patient_LGG_dir, volume_name)
        #patient_sp_dir1 = os.path.join(data_root, 'train_names_all_lggx3', volume_name1)
        #patient_sp_dir2 = os.path.join(data_root, 'train_names_all_lggx3', volume_name2)
       
        # for nifty
        if('nii' in file_postfix):
            if(mod + '.' in patient_sp_dir):
               if patient_sp_dir.find("_SP_") != -1 or patient_sp_dir.find("Label") != -1: # superpixel filter                    print('patient_sp_dir: ', patient_sp_dir)
                    save_array_as_nifty_volume(sp_image,patient_sp_dir)        
                #save_array_as_nifty_volume(sp_image,patient_sp_dir1)   
                #save_array_as_nifty_volume(sp_image,patient_sp_dir2)   
        # for mha
        else:
            img_file_dirs = os.listdir(patient_dir)        
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + file_postfix
                    if not os.path.exists(os.path.join(patient_sp_dir, img_file_dir)):
                        os.makedirs(os.path.join(patient_sp_dir, img_file_dir))
                    break
            assert(volume_name is not None)
            volume_name = os.path.join(patient_sp_dir, volume_name)
            save_array_volume_as_mha(volume_name, sp_image)    

def separate_volume(volume):
    patient_name, modality_postfix, label_postfix, data_root, file_postfix = volume

    data_num = len(patient_name)
    for i in range(data_num) :
        volume_list = []
        #print("i", patient_name[i])
        for _, mod_postfix in enumerate(modality_postfix) :            
            volume, volume_name = load_one_volume(patient_name[i], mod_postfix, data_root, file_postfix)
            #print("volumename: ",volume_name)
            save_separate(patient_name[i], volume_name, mod_postfix, volume, data_root, file_postfix)
        
        label, labelname = load_one_volume(patient_name[i], label_postfix, data_root, file_postfix)
        #print("label: ", labelname)
        save_separate(patient_name[i], labelname, label_postfix, label, data_root, file_postfix)

def process_volume(volume):

    patient_name, modality_postfix, data_root, file_postfix = volume
    
    n_supix=10
    data_num = len(patient_name)
    for i in range(data_num) :
        volume_list = []
        #print("i", patient_name[i])
        for _, mod_postfix in enumerate(modality_postfix) :           
            volume, volume_name = load_one_volume(patient_name[i], mod_postfix, data_root, file_postfix)
            volume_list.append(volume)

        mod_volumes = np.stack((volume_list[0],volume_list[1],volume_list[2],volume_list[3]),axis=-1)
        #print("mod_volumes: ",mod_volumes.shape)
        volume = volume_list[0]
        img_edges = np.zeros_like(volume, dtype=np.float32)
        for n_depth in range(volume.shape[2]):
            img_edge = skimage.feature.canny(volume[:,:,n_depth], sigma=2., low_threshold=0.1,
                high_threshold=0.2)
            img_edge = (img_edge * 255).astype(np.float32)             
            img_edges[:, :, n_depth] = img_edge
        #print("edges ", img_edges.shape)
        #print("mod_volumes ", mod_volumes.shape)
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        bosupix.build_3d(mod_volumes, img_edges)
        sp_image = bosupix.average(n_supix, 4, mod_volumes)
        #print("sp_image shape: ",sp_image.shape)
        #nda = np.transpose(nda, (0, 1, 2, 3))
        #print("sp_image shape: ",sp_image.shape)
        # nda = nda[:, :, :, 0]
        #print("mod {}",mod_postfix)
        save_superpixel(patient_name[i], modality_postfix, nda, data_root, file_postfix)   
        

class DataSuperpixel():
    def __init__(self, config):
        """
        Initialize the calss instance
        inputs:
            config: a dictionary representing parameters
        """
        self.config    = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [config['data_root']]
        self.sp_path = os.path.join(self.data_root[0], 'superpixel')
        if not os.path.exists(self.sp_path):
            os.makedirs(self.sp_path)
        self.modality_postfix     = config.get('modality_postfix', ['flair','t1', 't1ce', 't2'])
        self.intensity_normalize  = config.get('intensity_normalize', [True, True, True, True])
        self.with_ground_truth    = config.get('with_ground_truth', False)
        self.label_convert_source = config.get('label_convert_source', None)
        self.label_convert_target = config.get('label_convert_target', None)
        self.label_postfix = config.get('label_postfix', 'seg')
        self.file_postfix  = config.get('file_postfix', 'nii.gz')
        self.data_names    = config.get('data_names', None)
        self.data_num      = config.get('data_num', None)
        self.data_resize   = config.get('data_resize', None)
        self.with_flip     = config.get('with_flip', False)

        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
            
    def __get_patient_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient 
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        if(self.data_names is not None):            
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]
        # use all the patient names in data_root
        else:
            patient_names = os.listdir(self.data_root[0])
            patient_names = [name for name in patient_names if 'brats' in name.lower()]
        return patient_names
   

    def __load_one_volume(self, patient_name, mod):
        patient_dir = os.path.join(self.data_root[0], patient_name)
        # for bats17
        if('nii' in self.file_postfix):
            image_names = os.listdir(patient_dir)
            volume_name = None
            for image_name in image_names:
                if(mod + '.' in image_name):
                    volume_name = image_name                    
                    break
        # for brats15
        else:
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + self.file_postfix                    
                    break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_dir, volume_name)
        volume = load_3d_volume_as_array(volume_name)
        return volume, volume_name


    def preprocess_data(self):
        """
        load all the training/testing data
        """
        self.patient_names = self.__get_patient_names()
        assert(len(self.patient_names)  > 0)
        
        data_num = self.data_num if (self.data_num is not None) else len(self.patient_names)
        print("len: ",len(self.modality_postfix))
        volumes = (self.patient_names, self.modality_postfix, self.label_postfix, self.data_root[0], self.file_postfix) 
        #volumes = [self.patient_names[p] for p in range(data_num)]
        separate_volume(volumes)
        #process_volume(volumes)
   
    def get_total_image_number(self):
        """
        get the toal number of images
        """
        return len(self.data)
    
    def get_image_data_with_name(self, i):
        """
        Used for testing, get one image data and patient name
        """
        return [self.data[i], self.weight[i], self.patient_names[i], self.image_names[i], self.bbox[i], self.in_size[i]]


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))

    config = parse_config(config_file)
    config_data  = config['data']

    print(config_data)
    ds = DataSuperpixel(config_data)
    ds.preprocess_data()