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
from sklearn import preprocessing
import datetime

# creating an image reader.
# creating an image reader.
data_param = \
     {
      'CT': {'path_to_search': '~/niftynet/data/DB/CT'},
      'MR': {'path_to_search': '~/niftynet/data/DB/MR'}          
     } 

reader = ImageReader().initialise(data_param)

print(reader.shapes)
print(reader.tf_dtypes)
print(reader._dtypes)
print("num_subject:",reader.num_subjects)
#c = np.concatenate((a,b),axis=2)

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def slicer(str):
    for k in range(reader.num_subjects) :
        idx, image_data, interp_order = reader(idx=k)
        print("inter: ",interp_order)
        

        foldername = "/home/gergely/niftynet/data/{}slices{}/".format(str,k)
        for i in range(image_data[str].shape[2]):
            imb = image_data[str][:,:,i,0,0]

            if imb.shape[0] != 288 : 
                image_rescaled = cv2.resize(imb,(288,288),cv2.INTER_AREA)
                print("rescaled: ",image_rescaled.shape)
                imb=image_rescaled      

            C0 = imb.mean()
            C1 = imb.std()
            imb = (imb-C0)/C1

            print("imb avg: {},std: {}", imb.mean(), imb.std())
            #suffix = id_generator(6,"CHAWXSMNO")            
            #basename = "brain"
            #suffix = datetime.datetime.now().strftime("%H%M%S")
            #filename = str(uuid.uuid4()
            #filename = "{}.nii.gz".format(suffix)  
            filename = "{}Fig00{}.nii.gz".format(k,i)  
            misc.save_volume_5d(np.asarray(imb, dtype=np.float32),filename,foldername)

slicer('MR')
slicer('CT')