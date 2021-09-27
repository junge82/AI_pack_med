#!/usr/bin/env python3

# standard lib
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
    
import argparse

# numpy family
import numpy as np

# 3rd party
import skimage
import skimage.feature
import cv2
from multiprocessing import Pool
from readFlowFile import read




def boruvkasupix3D(params):

    root_folder, root_of_folder, scene_id = params
    infolder = join(root_folder, scene_id)
    offolder = join(root_of_folder, scene_id)

    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    onlyfiles.sort()

    img_in = skimage.io.imread(join(infolder, onlyfiles[0]))

    of_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], 2), dtype=np.int16)
    
    for i, f in enumerate(onlyfiles):
        if i < len(onlyfiles) - 1:
            of_file = join(offolder, f[:-3]+ 'flo') 
            of = read(of_file)

            #of = of * of_lmb

            of_ins[i, :, :, :] = of

    mn = np.amin(of_ins)
    mx = np.amax(of_ins)

    m = np.mean(np.absolute(of_ins))
    amax = np.amax(np.absolute(of_ins))


    print(scene_id, root_folder, round(m*100/img_in.shape[0], 2), round(amax*100/img_in.shape[0], 2))
    
    

def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image folder with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('root_folder',
            help='input0 image folder')
    parser.add_argument('root_of_folder',
            help='number of superpixels')
    args = parser.parse_args(argv)
    return args

def process_folders(root_folder, root_of_folder):
    
    params = []
    scene_ids = listdir(root_folder)
    scene_ids.sort()
    for scene_id in scene_ids:
        params.append((root_folder, root_of_folder, scene_id))


    #for p in params:
    #    boruvkasupix3D(p)
    pool = Pool(processes=10) 
    pool.map(boruvkasupix3D, params)


def main():
    args = parse_arguments(sys.argv[1:])
    process_folders(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
