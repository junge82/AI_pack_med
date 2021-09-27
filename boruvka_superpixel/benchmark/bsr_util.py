#!/usr/bin/env python3

"""Support for Berkeley Segmentation Resources"""

# stdlib
import os
from pathlib import Path

# 3rd party
import skimage.io
import scipy.io

prefix = os.path.expanduser('~/data/BSR')
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz'

class Error(Exception): pass

_ranges = {
        'train':    200,
        'val':      100,
        'test':     200,
        'all':      500,
        }

def assert_data_presence():
    img_dir = Path(prefix) / 'BSDS500' / 'data' / 'images' / 'train'
    if not img_dir.is_dir():
        msg = ('BSR data cannot be found at ' + prefix
                + '\ndownload from ' + url)
        raise Error(msg)


def img_range(split='all'):
    return range(_ranges[split])

def _canonical(num, split):
    assert num >= 0, 'illegal num'
    if split == 'all':
        if num < 200:
            split = 'train'
        elif num < 300:
            num -= 200
            split = 'val'
        else:
            num -= 300
            split = 'test'
    assert num < _ranges[split], 'illegal num'
    return num, split

def img_path(num, split='all'):
    assert_data_presence()
    num, split = _canonical(num, split)
    img_dir = Path(prefix) / 'BSDS500' / 'data' / 'images' / split
    paths = list(img_dir.glob('*.jpg'))
    paths.sort()
    assert len(paths) == _ranges[split]
    return str(paths[num])

def imread(num, split='all'):
    return skimage.io.imread(img_path(num, split))

def seg_read(num, split='all'):
    assert_data_presence()
    num, split = _canonical(num, split)
    seg_dir = Path(prefix) / 'BSDS500' / 'data' / 'groundTruth' / split
    paths = list(seg_dir.glob('*.mat'))
    paths.sort()
    assert len(paths) == _ranges[split]
    data = scipy.io.loadmat(str(paths[num]))['groundTruth']
    assert data.shape[0] == 1
    res = []
    for i in range(data.shape[1]):
        seg = data[0,i][0,0][0]
        assert(seg.min() == 1)
        res.append(seg - 1)
    return res


def test_seg():
    for i in img_range():
        print(i, seg_read(i).shape)


# vim: set sw=4 sts=4 expandtab :
