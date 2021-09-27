#!/usr/bin/env python3

# standard lib
import sys
import argparse

# numpy family
import numpy as np

# 3rd party
import skimage
import skimage.feature

# local

sys.path.insert(0, "../pybuild")
import boruvka_superpixel

def boruvkasupix(infile, outfile, n_supix):
    img_in = skimage.io.imread(infile)
    img_gray = skimage.color.rgb2gray(img_in)
    img_edge = skimage.feature.canny(img_gray, sigma=2., low_threshold=0.1,
            high_threshold=0.2)
    img_edge = (img_edge * 255).astype(np.uint8)
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_2d(img_in, img_edge)
    out = bosupix.average(n_supix, 3, img_in)
    skimage.io.imsave(outfile, out)


def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile',
            help='input image file')
    parser.add_argument('outfile',
            help='output image file')
    parser.add_argument('n_supix',
            type=int,
            help='number of superpixels')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_arguments(sys.argv[1:])
    boruvkasupix(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
