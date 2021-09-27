#!/usr/bin/env python3

# standard lib
import sys
import argparse

# numpy family
import numpy as np
from scipy.ndimage import sobel, gaussian_filter

# 3rd party
import skimage
import skimage.feature

# local
sys.path.append('../pybuild')
import boruvka_superpixel
import bsr_util
#import benchmarks
import pyximport ; pyximport.install()
import benchmarks_cy as benchmarks

def get_data(data):
    if data in ('train', 'val', 'test', 'all'):
        for num in bsr_util.img_range(data):
            img = bsr_util.imread(num, data)
            seg = bsr_util.seg_read(num, data)
            yield img, seg
    else:
        try:
            n = int(data)
            assert 1 <= n <= 500
        except (ValueError, AssertionError) as e:
            raise ValueError('when data is number N:  1 <= N <= 500')
        for num in range(n):
            img = bsr_util.imread(num, 'all')
            seg = bsr_util.seg_read(num, 'all')
            yield img, seg

def do_asa(data):
    n_supix = 100
    asa_arr = []
    for i, (img_in, seg) in enumerate(get_data(data)):
        # input image
        #img_dat = img_in.astype(float)          # rgb
        #img_dat = skimage.color.rgb2hsv(img_in) # hsv
        img_dat = skimage.color.rgb2lab(img_in) # Lab
        # edge strength
        img_gray = skimage.color.rgb2gray(img_in)
        #img_edge = skimage.feature.canny(img_gray, sigma=2., low_threshold=0.1,
        #        high_threshold=0.2).astype(float)
        img_edge = np.sqrt(sobel(img_gray, axis=0)**2
                + sobel(img_gray, axis=1)**2)
        img_edge = gaussian_filter(img_edge, 3.)
        img_edge *= 1
        # supix
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        bosupix.build_2d(img_dat, img_edge)
        supix = bosupix.label(n_supix)
        asa_i_arr = []
        for segment in seg:
            asa_i_arr.append(benchmarks.asa(supix, segment))
        asa_i_mean = np.array(asa_i_arr).mean()
        print('{:3d}: {} x {}: {}  {}'.format(i,
            img_in.shape[0], img_in.shape[1], len(seg), asa_i_mean))
        asa_arr.append(asa_i_mean)
    print('avg: {:.3f}'.format(np.array(asa_arr).mean()))


def do_benchmark(test_name, data):
    if test_name == 'asa':
        do_asa(data)
    else:
        raise ValueError('unknown test_name:', test_name)

def parse_arguments(argv):
    description = ('Test 2D boruvka_superpixel with BSDS500 dataset.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('test_name',
            choices=('asa',),
            help='which test')
    parser.add_argument('--data',
            default='all',
            help='select data: train | val | test | all | N, '
            'where 1 <= N <= 500  [default: %(default)s]')
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_arguments(sys.argv[1:])
    do_benchmark(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
