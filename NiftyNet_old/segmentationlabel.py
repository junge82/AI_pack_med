
from niftynet.io.image_reader import ImageReader
import tensorflow as tf
import numpy as np
""" import paramiko
host = "nipg11.inf.elte.hu"
port = 10113
transport = paramiko.Transport((host, port))


transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)

filepath = '/media/hdd/Meddatabase/BRATS2015_VSD/Dataset_NiftyNet/Brats15_renamed_training/HGG'
localpath = '/home/remotepasswd'
sftp.get(filepath, localpath)
 """
# data_param = \
#      {
#       'label': {'path_to_search': '/media/hdd/Meddatabase/BRATS2015_VSD/Dataset_NiftyNet/Brats15_renamed_training/HGG',
#       #~/niftynet/data/BRATS_examples/HGG',
#                 'filename_contains': 'Label',
#                 }      
#      }

# data_param = {'T1':    {'path_to_search': '~/niftynet/data/BRATS_examples/HGG',
#                         'filename_contains': 'T1', 'filename_not_contains': 'T1c'},
#               'T1c':   {'path_to_search': '~/niftynet/data/BRATS_examples/HGG',
#                         'filename_contains': 'T1c'},
#               'T2':    {'path_to_search': '~/niftynet/data/BRATS_examples/HGG',
#                         'filename_contains': 'T2'},
#               'Flair': {'path_to_search': '~/niftynet/data/BRATS_examples/HGG',
#                         'filename_contains': 'Flair'},
#               'label': {'path_to_search': '~/niftynet/data/BRATS_examples/HGG',
#                         'filename_contains': 'Label'}} 


data_param = {'T1':    {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'T1', 'filename_not_contains': ('T1c','SP_T1')},
              'T1c':   {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'T1c'},
              'T2':    {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'T2'},
              'Flair': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'Flair'},
              'SP_T1': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'SP_T1', 'filename_not_contains': 'SP_T1c'},
              'SP_T1c': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'SP_T1c'},
              'SP_T2': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'SP_T2'},
              'SP_Flair': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'SP_Flair'},
              'label': {'path_to_search': '/home/gergely/Dataset/HGG',
                        'filename_contains': 'Label'}} 

                        

# {'csv_file', 'path_to_search',
#  'filename_contains', 'filename_not_contains',
#  'interp_order', 'pixdim', 'axcodes', 'spatial_window_size',
#  'loader'}

grouping_param = {'image': ('T1', 'T1c', 'T2', 'Flair', 'SP_T1', 'SP_T1c', 'SP_T2', 'SP_Flair'), 'label':('label',)}
reader = ImageReader().initialise(data_param, grouping_param)                            


idx, image_data, interp_order = reader(idx=0)
print("image: {}, label: {}",image_data['image'].shape,image_data['label'].shape)
print("len: ",len(np.unique(image_data['label'])))

# reader as a generator
def image_label_pair_generator():
    """
    A generator wrapper of an initialised reader.
    
    :yield: a dictionary of images (numpy arrays).
    """
    while True:
        _, image_data, _ = reader()
        yield image_data

# tensorflow dataset
dataset = tf.data.Dataset.from_generator(
    image_label_pair_generator,
    output_types=reader.tf_dtypes)
    #output_shapes=reader.shapes)
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()

# run the tensorlfow graph
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for _ in range(3):
        data_dict = sess.run(iterator.get_next())
        print(data_dict.keys())
        print('image: {}, label: {}'.format(
            data_dict['image'].shape,
            data_dict['label'].shape))
# sftp.close()
# transport.close()

from niftynet.layer.rand_rotation import RandomRotationLayer as Rotate

data_param = {'MR': {'path_to_search': '~/niftynet/data/BRATS_examples/HGG'}}
reader = ImageReader().initialise(data_param)

_, image_data, _ = reader(idx=0)
import matplotlib.pyplot as plt

plt.imshow(image_data['MR'][:, :, 40, 0, 0])
plt.show()

rotation_layer = Rotate()
rotation_layer.init_uniform_angle([-10.0, 10.0])
reader.add_preprocessing_layers([rotation_layer])

_, image_data, _ = reader(idx=0)
image_data['MR'].shape

plt.imshow(image_data['MR'][:, :, 40, 0, 0])
plt.show()

