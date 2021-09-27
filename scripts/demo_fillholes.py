from matplotlib.pyplot import *
import skimage
import skimage.io
import skimage.measure
import niftynet.io.misc_io as misc

img_orig = skimage.io.imread('/home/gergely/niftynet/data/BiomedJournal/MRmask9/new_maszk0.png')

# min and max values, unfortunately not 2D 0 and 1
print(img_orig.shape, img_orig.min(), img_orig.max())

# create bool image
tmp = img_orig.sum(axis=2)
imb = tmp > tmp.mean()
print(imb.shape, imb.min(), imb.max())
# probably this is what you get from the niftynet mask generator

# remove all clusters except largest
#
# labels: background=0, all clusters get distict integer value
labels = skimage.measure.label(imb)
# histo: size of clusters (as well as background)
histo = skimage.exposure.histogram(labels)[0]
# size of largest non-background cluster
largest = histo[1:].max()
# remove all others
for i in range(1, labels.max()+1):
	if histo[i] != largest:
		imb[labels == i] = 0
imshow(imb)

# remove holes
#
# labels: now mask=0, background clusters get disticts integer
labels = skimage.measure.label(1 - imb)
# fill in all now-background clusters, except the one on the outside
# (ie. containing 0,0)
for i in range(1, labels.max()+1):
	if labels[0,0] != i:
		imb[labels == i] = 1
b = np.asarray(imb, dtype=np.float32)
misc.save_volume_5d(b,"holehole","/home/gergely/Asztal")
skimage.io.imshow(imb)



