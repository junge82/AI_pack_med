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

# local

sys.path.insert(0, "../pybuild")
import boruvka_superpixel

def boruvkasupix2D(infolder, outfolder, n_supix):

    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    onlyfiles.sort()

    if not exists(outfolder):
        makedirs(outfolder)
    
    for i, f in enumerate(onlyfiles):
        img_in = skimage.io.imread(join(infolder, f))
        img_gray = skimage.color.rgb2gray(img_in)
        img_edge = skimage.feature.canny(img_gray, sigma=2., low_threshold=0.1,
            high_threshold=0.2)
        img_edge = (img_edge * 255).astype(np.uint8)
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        bosupix.build_2d(img_in, img_edge)

        label = bosupix.label(n_supix)
        print(label.shape)

        #out = bosupix.average(n_supix, 3, img_in)

        skimage.io.imsave(join(outfolder, f), (label*20).astype(np.uint8))
        print(i)
    
    



def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image folder with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infolder',
            help='input image folder')
    parser.add_argument('outfolder',
            help='output image folder')
    parser.add_argument('n_supix',
            type=int,
            help='number of superpixels')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_arguments(sys.argv[1:])
    boruvkasupix2D(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
