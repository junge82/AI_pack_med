import numpy as np
import  niftynet.io.image_loader as ImageLoader
import nibabel as nib
from matplotlib.pyplot import *
from nibabel.testing import data_path
import scipy
import skimage
from skimage import filters
import os
import matplotlib.pyplot as plt 

example_ni1 = os.path.join(data_path, '/home/gergely/NiftyNet/data/mr_ct_regression/T2_corrected/CHA.nii.gz')
n1_img = nib.load(example_ni1)
print(n1_img.shape)
#img_nii = nib.Nifti1Image(n1_img, t1_affine)
nib.save(n1_img, "kecske")

param = 10
file1 = ImageLoader.load_image_obj('/home/gergely/NiftyNet/data/mr_ct_regression/T2_corrected/CHA.nii.gz')
t1_affine = file1.affine
img = file1.dataobj[:,:]
imshow(img.T)
imsave("akarmi.jpg",img)
plt.imshow(img.T)
plt.show()

matplotlib.pyplot.imshow(img.T)
file2 = ImageLoader.load_image_obj('/home/gergely/NiftyNet/data/mr_ct_regression/T2_mask/CHA.nii.gz')
mask_orig = file2.dataobj[:,:]
affine_m= file2.affine
plt.imshow(mask_orig.T)
plt.show()
img_nii = nib.Nifti1Image(mask_orig,affine_m)
nib.save(img_nii, "maszk")




# intersection over union
# a standard way to quantify how two pixelsets differ
# disjunct: 0   perfect overlap: 1
def iou(a, b):
	intersec = np.logical_and(a.astype(bool), b.astype(bool))
	union = np.logical_or(a.astype(bool), b.astype(bool))
	return intersec.sum() / union.sum()


thres = filters.threshold_otsu(img)
img2 = 1 - (img > thres)
# imshow(img2.T)
plt.imshow(img2.T)
plt.show()

img3 = scipy.ndimage.distance_transform_edt(img2)
#imshow(img3.T)
plt.imshow(img3.T)
plt.show()

# expanded by param
img4 = 1 - (img3 > param)
#imshow(img4.T)
plt.imshow(img4.T)
plt.show()
print(iou(img4, mask_orig))	# not too high

img5 = scipy.ndimage.distance_transform_edt(img4)
#imshow(img5.T)
plt.imshow(img5.T)
plt.show()

# now shrinked by param
img6 = 1 - (img5 < param)
#imshow(img6.T)
plt.imshow(img6.T)
plt.show()
print(iou(img6, mask_orig))	# better


# hand tuning...
img7 = 1 - (img5 < 8.5)
#imshow(img7.T)
plt.imshow(img7.T)
plt.show()
print(iou(img7, mask_orig))	# almost perfect


imsave("maszk.jpg",img7.T)
img_nii = nib.Nifti1Image(img7.T,affine_m)
nib.save(img_nii, "maszk")

plt.clf()
ax = plt.imshow(img7.T)
plt.show()
plt.colorbar(ax)



