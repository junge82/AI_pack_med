from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
import nibabel as nib
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import *
import niftynet.io.misc_io as misc
import skimage
import skimage.io
import skimage.measure
from skimage.transform import rescale, resize, downscale_local_mean

# creating an image reader.
# creating an image reader.
data_param = \
    {'CT': {'path_to_search': '~/niftynet/data/DB'}          
    } 
reader = ImageReader().initialise(data_param)

print(reader.shapes)
print(reader.tf_dtypes)
print(reader._dtypes)
print("num_subject:",reader.num_subjects)

for k in range(reader.num_subjects) :
    idx, image_data, interp_order = reader(idx=k)
    print("inter: ",interp_order)
    slice_num = image_data['CT'].shape[2]  
    
    foldername = "/home/gergely/niftynet/data/slices{}/".format(k)
    for i in range(image_data['CT'].shape[2]):
        
        imb = image_data['CT'][:,:,i,0,0]
    
        if(imb.shape[0]!=256):

            imb = imb - imb.mean()
            imb = imb * (-1/imb.min())
            imb = imb * (1/imb.max())
            #image_rescaled = rescale(imb, skala, anti_aliasing=False)
            image_rescaled = resize(imb, (256, 256), anti_aliasing=True)
            imb = image_rescaled
            print("rescaled: ",image_rescaled.shape)
            
        
        imb = imb - imb.mean()
        imb = imb * (-0.8/imb.min())
        filename = "{}Fig00{}.nii.gz".format(k,i)  
        misc.save_volume_5d(np.asarray(imb, dtype=np.float32),filename,foldername)
        
   


