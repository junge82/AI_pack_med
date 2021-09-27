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

# local

sys.path.insert(0, "../pybuild")
import boruvka_superpixel

def boruvkasupix2D_3D(imfolder, sp2dfolder, sp3dfolder, out_video):

    onlyfiles = [f for f in listdir(imfolder) if isfile(join(imfolder, f))]
    onlyfiles.sort()
    img_in = skimage.io.imread(join(imfolder, onlyfiles[0]))
    w, h, d = img_in.shape
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(out_video,fourcc, 20.0, (h, w*3))
    for f in onlyfiles:
        img_in = skimage.io.imread(join(imfolder, f))
        sp2d_in = skimage.io.imread(join(sp2dfolder, f))
        sp3d_in = skimage.io.imread(join(sp3dfolder, f))

        frame = np.concatenate((img_in, sp2d_in, sp3d_in))
        out.write(frame)       
    
    out.release()

    


def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image folder with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('imfolder',
            help='input image folder')
    parser.add_argument('sp2dfolder',
            help='output image folder')
    parser.add_argument('sp3dfolder',
            help='output image folder')
    parser.add_argument('out_video',
            help='number of superpixels')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_arguments(sys.argv[1:])
    boruvkasupix2D_3D(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
