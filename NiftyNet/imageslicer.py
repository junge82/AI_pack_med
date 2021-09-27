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
import cv2
import tensorflow as tf
from skimage.transform import rescale, resize, downscale_local_mean

# creating an image reader.
# creating an image reader.
data_param = \
     {'CT': {'path_to_search': '~/niftynet/data/DB/CT'},
      'MR': {'path_to_search': '~/niftynet/data/DB/MR'}          
     } 


reader = ImageReader().initialise(data_param)

print(reader.shapes)
print(reader.tf_dtypes)
print(reader._dtypes)
print("num_subject:",reader.num_subjects)
#c = np.concatenate((a,b),axis=2)

def slicer(str):
    for k in range(reader.num_subjects) :
        idx, image_data, interp_order = reader(idx=k)
        print("inter: ",interp_order)

        C0 = image_data[str][:,:,:,0,0].mean()
        C1 = image_data[str][:,:,:,0,0].std()
        print("image_data avg: {},std: {}", C0, C1) 

        foldername = "/home/gergely/niftynet/data/{}slices{}/".format(str,k)
        for i in range(image_data[str].shape[2]):
            imb = image_data[str][:,:,i,0,0]

            if(imb.shape[0]!=288):

                imb = cv2.resize(imb, (288, 288), cv2.INTER_AREA)
                print("__rescaled__")
                print("imb avg: {},std: {}", imb.mean(), imb.std())

            # imb = imb - imb.mean()
            # imb = imb * (-0.8/imb.min())
            # #imb = imb - imb.std()

            filename = "{}Fig00{}.nii.gz".format(k,i)  
            misc.save_volume_5d(np.asarray(imb, dtype=np.float32),filename,foldername)

slicer('MR')
slicer('CT')