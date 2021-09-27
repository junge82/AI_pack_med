import numpy as np
import nibabel as nib
from nibabel.testing import data_path
from matplotlib.pyplot import *
import scipy
import skimage
from skimage import filters
import os
import sys

# intersection over union
# a standard way to quantify how two pixelsets differ
# disjunct: 0   perfect overlap: 1
def iou(a, b):
	intersec = np.logical_and(a.astype(bool), b.astype(bool))
	union = np.logical_or(a.astype(bool), b.astype(bool))
	return intersec.sum() / union.sum()

param = 10
n_img_orig = os.path.join(data_path,'/home/gergely/NiftyNet/data/mr_ct_regression/T2_corrected/CHA.nii.gz')
n_img_orig_mask = os.path.join(data_path,'/home/gergely/NiftyNet/data/mr_ct_regression/T2_mask/CHA.nii.gz')

new_img_orig = os.path.join(data_path,'/home/gergely/niftynet/data/BiomedJournal/MR/Fig8_MRI_Registered.nii')
 
n_img = nib.load(n_img_orig)
print(n_img.shape)
n_img_mask = nib.load(n_img_orig_mask)
print(n_img_mask.shape)

new_img = nib.load(new_img_orig)
print(new_img.shape)
data_new_img = new_img.dataobj[:,:,0,0]
print(data_new_img.shape)

affine_m = n_img_mask.affine
mask_orig =  n_img_mask.dataobj[:,:]


def masking(img,z):
   
 try:
    thres = skimage.filters.threshold_otsu(img)
    img2 = 1 - (img > thres)

    img3 = scipy.ndimage.distance_transform_edt(img2)

    # expanded by param
    img4 = 1 - (img3 > param)

    # print(iou(img4, mask_orig))	# not too high

    img5 = scipy.ndimage.distance_transform_edt(img4)

    # now shrinked by param
    img6 = 1 - (img5 < param)
    #print(iou(img6, mask_orig))	# better


    # hand tuning...
    img7 = 1 - (img5 < 8.5)
    #print(iou(img7, mask_orig))	# almost perfect

    str = "/home/gergely/niftynet/data/BiomedJournal/MRmask/new_maszk{}.png".format(z)
    imsave(str,img7.T)

    img_nii = nib.Nifti1Image(img7.T,None)
    str2 = "maszk{}.nii".format(z)
    nib.save(img_nii, str2)
  #      nib.save(img_nii, os.path.join('build',str2))
    
 except ValueError:
     print("ValueError")
     pass
 except:
     print("Exception ",sys.exc_info()[0])
     pass


print(new_img.dataobj.shape[2])
for i in range(new_img.dataobj.shape[2]): 
    temp   = new_img.dataobj[:,:,i,0]    
    masking(temp,i)