#!/usr/bin/env python3

# standard lib
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
    
import argparse
from readFlowFile import read

# numpy family
import numpy as np

# 3rd party
import skimage
import skimage.feature
from scipy.ndimage import sobel, gaussian_filter
import cv2

# local

sys.path.insert(0, "../pybuild")
import boruvka_superpixel

import colorsys
from multiprocessing import Pool

def get_random_color():
    return list(np.random.choice(range(256), size=3))

def random_colors(N):
    colors = np.unique([get_random_color() for _ in range(N)], axis=0).tolist()
    while len(colors) < N:
        colors += [get_random_color() for _ in range(N - len(colors))]
        colors = np.unique(colors, axis=0).tolist()
    return colors

def average(data, label):
    
    sp_ids = np.unique(label)
    avg = np.zeros_like(data)

    for sp_id in sp_ids:
        avg[label==sp_id] = np.mean(data[label==sp_id], axis=0)
    return avg

def boruvkasupix3D(params):

    root_folder, root_depth_folder, scene_id, outfolder, n_supix, depth_prefactor = params
    infolder = join(root_folder, scene_id)
    depthfolder = join(root_depth_folder, scene_id)

    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    onlyfiles.sort()
    img_in = skimage.io.imread(join(infolder, onlyfiles[0]))
    img_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], img_in.shape[2]+1), dtype=np.uint8)
    img_grey_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1]), dtype=np.uint8)
    img_edges = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1]), dtype=np.uint8)
    for i, f in enumerate(onlyfiles):
        img_in = skimage.io.imread(join(infolder, f))
        depth_in = skimage.io.imread(join(depthfolder, f))
        depth_in = skimage.transform.resize(depth_in, (img_in.shape[0], img_in.shape[1]), anti_aliasing=True)
        depth_in = skimage.img_as_ubyte(depth_in)
        depth_in = np.round(depth_prefactor * depth_in).astype(img_ins.dtype)
        img_ins[i, :, :, :] = np.concatenate([img_in, depth_in[:, :, None]], axis=-1)
        img_gray = skimage.color.rgb2gray(img_in)
        img_grey_ins[i, :, :] = (img_gray * 255).astype(np.uint8)
        # img_edge = np.sqrt(sobel(img_gray, axis=0)**2
        #         + sobel(img_gray, axis=1)**2)
        # img_edge = gaussian_filter(img_edge, 3.)
        # img_edge *= 1
        # img_edge = (img_edge * 255).astype(np.uint8)
        # img_edges[i, :, :] = img_edge
        

    img_ins = np.transpose(img_ins, (1, 2, 0, 3))
    img_grey_ins = np.transpose(img_grey_ins, (1, 2, 0))
    img_edges = np.transpose(img_edges, (1, 2, 0))

    label_outs = np.zeros((img_in.shape[0], img_in.shape[1], len(onlyfiles)), dtype=np.int16)

    
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_3d(img_ins, img_edges)
    bosupix.label_o(n_supix, label_outs)
    colors = random_colors(n_supix)
    img_ins = img_ins[:, :, :, :3]
    avg_outs = average(img_ins, label_outs)
    avg_grey_outs = average(img_grey_ins, label_outs)
    
    
    rnd_img_outs = np.zeros((img_in.shape[0], img_in.shape[1], len(onlyfiles), img_in.shape[2]), dtype=np.uint8)
    for i in range(n_supix):
        rnd_img_outs[label_outs == i] = colors[i]
    
       
    label_out_folder = join(outfolder, 'label')
    rnd_out_folder = join(outfolder, 'rnd')
    avg_out_folder = join(outfolder, 'avg')
    avg_grey_out_folder = join(outfolder, 'avg_grey')
    

    for folder in [outfolder, label_out_folder, rnd_out_folder, avg_grey_out_folder, avg_out_folder]:
        if not exists(folder):
            makedirs(folder)

    w, h, d = img_in.shape
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_video = join(outfolder, '{}_3d_{}.mp4'.format(scene_id, n_supix))
    vid_out = cv2.VideoWriter(out_video,fourcc, 20.0, (h*2, w*2))

    
    for i, f in enumerate(onlyfiles):

        skimage.io.imsave(join(label_out_folder, f[:-3]+'png'), label_outs[:, :, i])
        skimage.io.imsave(join(rnd_out_folder, f[:-3]+'png'), rnd_img_outs[:, :, i, :])
        skimage.io.imsave(join(avg_out_folder, f[:-3]+'png'), avg_outs[:, :, i, :])
        skimage.io.imsave(join(avg_grey_out_folder, f[:-3]+'png'), avg_grey_outs[:, :, i])

             

        frame_rnd = np.concatenate((img_ins[:, :, i, :], rnd_img_outs[:, :, i, :]), axis=1)

        grey_3_channel = skimage.color.gray2rgb(avg_grey_outs[:, :, i])
        frame_avg = np.concatenate((avg_outs[:, :, i, :], grey_3_channel), axis=1)

        frame = np.concatenate((frame_rnd,frame_avg))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid_out.write(frame)
    vid_out.release()
    print(scene_id)

def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image folder with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('root_folder',
            help='input image folder')
    parser.add_argument('root_depth_folder',
            help='input depth folder')
    parser.add_argument('out_root_folder',
            help='output image folder')
    parser.add_argument('sp_number',
            type=int,
            help='number of superpixels')
    parser.add_argument('depth_prefactor',
            type=float,
            help='depth prefactor')
    args = parser.parse_args(argv)
    return args

def process_folders(root_folder, root_depth_folder, out_root_folder, sp_number, depth_prefactor):
    if not exists(out_root_folder):
            makedirs(out_root_folder)
    params = []
    scene_ids = listdir(root_folder)
    scene_ids.sort()
    for scene_id in scene_ids:
        out = join(out_root_folder, "{}_3d_depth_{}_{}".format(scene_id, sp_number, int(depth_prefactor*100)))
        params.append((root_folder, root_depth_folder, scene_id, out, sp_number, depth_prefactor))


    #for p in params[:1]:
    #    boruvkasupix3D(p)
    pool = Pool(processes=8)
    pool.map(boruvkasupix3D, params)


def main():
    args = parse_arguments(sys.argv[1:])
    process_folders(**args.__dict__)

if __name__ == '__main__':
    sys.exit(main())

# vim: set sw=4 sts=4 expandtab :
