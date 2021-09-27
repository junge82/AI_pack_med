import os
import sys
import numpy as np

from multiprocessing import Pool

from subprocess import call

def process_image(paths):
    image_path = paths[0]
    out_path = paths[1]

    command = '../build/boruvkasupix image {} {} 800'.format(image_path, out_path)
    print(command)
    os.system(command)


images = []

root_path = '/home/fothar/cityscapes'

subsets = ["train", "val", "test"]

for subset in subsets:
    drive_image_dir = os.path.join(root_path, 'leftImg8bit', subset)

    drive_out_dir = os.path.join(root_path, 'superpixel', subset)

    if not os.path.exists(drive_out_dir):
        os.makedirs(drive_out_dir)

    drive_ids = os.listdir(drive_image_dir)

    for drive_id in drive_ids:       
            
        image_dir = os.path.join(drive_image_dir, drive_id)
        image_ids = ['_'.join(f.split('_')[:-1] ) for f in os.listdir(image_dir)]            
            

        
        out_dir = os.path.join(drive_out_dir, drive_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for image_id in image_ids:
            image_path=os.path.join(image_dir, "{}_leftImg8bit.png".format(image_id))
            out_path=os.path.join(out_dir, "{}_leftImg8bit.png".format(image_id))

            images.append((image_path, out_path))

            
                        
pool = Pool() 
pool.map(process_image, images)