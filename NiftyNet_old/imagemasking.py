from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
import nibabel as nib
import tensorflow as tf
import numpy as np
import niftynet.io.misc_io as misc
import skimage
import skimage.io
import skimage.measure
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import sys
# creating an image reader.
# creating an image reader.
data_param = \
    {
        'MR': {'path_to_search': '~/niftynet/data/DB/MR'},          
    } 

reader = ImageReader().initialise(data_param)

print(reader.shapes)
print(reader.tf_dtypes)
print(reader._dtypes)
print("num_subject:",reader.num_subjects)

def masker(str):
    for k in range(reader.num_subjects) :
        idx, image_data, interp_order = reader(idx=k)
        mask_layer = BinaryMaskingLayer(
                    type_str='mean_plus',
                    multimod_fusion='or',
                    threshold=1.0)

        mask_out = mask_layer(image_data['MR'])
    
        foldername = "/home/gergely/niftynet/data/{}mask{}/".format(str,k)
        foldername_out = "/home/gergely/niftynet/data/{}mask{}/out".format(str,k)
        for i in range(mask_out.shape[2]):
            imb = mask_out[:,:,i,0,0]
            try:
                # remove all clusters except largest
                #
                # labels: background=0, all clusters get distict integer value
                labels = skimage.measure.label(imb)
                # histo: size of clusters (as well as background)
                histo = skimage.exposure.histogram(labels)[0]
                # size of largest non-background cluster
                largest = histo[1:].max()
                # remove all others
                for j in range(1, labels.max()+1):
                    if histo[j] != largest:
                        imb[labels == j] = 0

                # remove holes
                #
                # labels: now mask=0, background clusters get disticts integer
                labels = skimage.measure.label(1 - imb)
                # fill in all now-background clusters, except the one on the outside
                # (ie. containing 0,0)
                for j in range(1, labels.max()+1):
                    if labels[0,0] != j:
                        imb[labels == j] = 1

                print("shape:",imb.shape[0])         
                if imb.shape[0] != 288 : 
                    #image_rescaled = resize(imb, (288, 288), anti_aliasing=False)                    
                    context = cv2.resize(np.array(imb, dtype=np.uint8),(288,288), cv2.INTER_AREA)
                    t2mask = cv2.resize(np.array(imb, dtype=np.uint8),(288,288), cv2.INTER_LINEAR)
                    print("context rescaled: ",context.shape)
                    print("t2mask rescaled: ",t2mask.shape)
                    
            
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise

            filename = "{}Fig00{}_niftynet_out.nii.gz".format(k,i) 
            #imsave(str,mask_out[:,:,i,0,0])   
            misc.save_volume_5d(np.asarray(context, dtype=np.float32),filename,foldername_out)
            filename = "{}Fig00{}.nii.gz".format(k,i) 
            #imsave(str,mask_out[:,:,i,0,0])   
            misc.save_volume_5d(np.asarray(t2mask, dtype=np.float32),filename,foldername)



masker('MR')