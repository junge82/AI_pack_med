#!/usr/bin/env python3

# standard lib
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
# numpy family
import numpy as np

# 3rd party
import skimage
import cv2

from multiprocessing import Pool
    
import argparse
from readFlowFile import read

sys.path.insert(0, "../pybuild")
import boruvka_superpixel
from config_3D_D import Config_3D_D
from boruvka_parameter import Boruvka3DOFParam, Boruvka3DOFReverseParam

config = Config_3D_D()


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

def boruvkasupix3D(param):

    of_reverse = param.NAME == "3DOF_REVERSE"
    struct_of = of_reverse or param.NAME == "3DOF"
    
    infolder = param.infolder()          

    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    onlyfiles.sort()
    print(infolder)
    img_in = skimage.io.imread(join(infolder, onlyfiles[0]))
    img_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], img_in.shape[2]), dtype=np.uint8)
    img_grey_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1]), dtype=np.uint8)
    img_edges = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1]), dtype=np.uint8)
    for i, f in enumerate(onlyfiles):
        img_in = skimage.io.imread(join(infolder, f))
        img_ins[i, :, :, :] = img_in
        img_gray = skimage.color.rgb2gray(img_in)
        img_grey_ins[i, :, :] = (img_gray * 255).astype(np.uint8)  

    for feature in param.FEATURES:
        if feature.NAME == ["OF"]:
            offolder = feature.featurefolder(param.SCENE_ID)
            max_of = img_in.shape[0] / 4
            of_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], 2), dtype=np.uint8)
            for i, f in enumerate(onlyfiles):
                
                if i < len(onlyfiles) - 1:
                    of_file = join(offolder, f[:-3]+ 'flo')
                    of = read(of_file)
                    of[of<-max_of] = -max_of
                    of[of>max_of] = max_of
                    of = 128 + feature.PREFACTOR * of/max_of * 127
                    of[of<0] = 0
                    of[of>255] = 255
                    of_ins[i, :, :, :] = of                
                else:
                    of_ins[i, :, :, :] = 128
            
            img_ins = np.concatenate([img_ins, of_ins], axis=-1)

        elif feature.NAME == ["DEPTH"]:
            depthfolder = feature.featurefolder(param.SCENE_ID)
            depth_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], 2), dtype=np.uint8)
            for i, f in enumerate(onlyfiles):
                depth_in = skimage.io.imread(join(depthfolder, f))
                depth_in = skimage.transform.resize(depth_in, (img_in.shape[0], img_in.shape[1]), anti_aliasing=True)
                depth_in = skimage.img_as_ubyte(depth_in)
                depth_in = np.round(feature.PREFACTOR * depth_in).astype(img_ins.dtype)
                depth_ins[i, :, :, :] = depth_in
                img_ins = np.concatenate([img_ins, depth_ins], axis=-1)
        
    img_ins = np.transpose(img_ins, (1, 2, 0, 3))
    img_grey_ins = np.transpose(img_grey_ins, (1, 2, 0))
    img_edges = np.transpose(img_edges, (1, 2, 0))    
    
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()


    if struct_of:
        offolder = param.offolder()  
        of_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], 2), dtype=np.int16)
        for i, f in enumerate(onlyfiles[:-1]):            
            of_file = join(offolder, f[:-3]+ 'flo')
            of = read(of_file)
            of = np.rint(of)
            of_ins[i, :, :, 0] = of[:, :, 1]
            of_ins[i, :, :, 1] = of[:, :, 0]

        of_ins = np.transpose(of_ins, (1, 2, 0, 3))

        if of_reverse:
            of_r_ins = np.zeros((len(onlyfiles), img_in.shape[0], img_in.shape[1], 2), dtype=np.int16)      
            ofreversefolder = param.ofreversefolder()
            for i, f in enumerate(onlyfiles[:-1]):                
                of_r_file = join(ofreversefolder, f[:-3]+ 'flo')
                ofr = read(of_r_file)
                ofr = np.rint(ofr)
                of_r_ins[i, :, :, 0] = ofr[:, :, 1]
                of_r_ins[i, :, :, 1] = ofr[:, :, 0]
        
            of_r_ins = np.transpose(of_r_ins, (1, 2, 0, 3))
            bosupix.build_3d_of2(img_ins, img_edges, of_ins, of_r_ins, param.OFEDGE_PREFACTOR, param.OF_TOLERANCE_SQ, param.OF_REL_TOLERANCE)
        else:   
            bosupix.build_3d_of(img_ins, img_edges, of_ins, param.OFEDGE_PREFACTOR)

    else:
        bosupix.build_3d(img_ins, img_edges)

    outfolder = param.out_folder()
    label_outs = np.zeros((img_in.shape[0], img_in.shape[1], len(onlyfiles)), dtype=np.int16)
        
    bosupix.label_o(param.SP_NUMBER, label_outs)

    save_result(img_ins[:, :, :, :3], img_grey_ins, label_outs, param.SP_NUMBER, outfolder, onlyfiles, config.SAVE_VIDEO)

def save_result(img_ins, img_grey_ins, label_outs, n_supix, outfolder, onlyfiles, save_vid):
    colors = random_colors(n_supix)
    avg_outs = average(img_ins, label_outs)
    avg_grey_outs = average(img_grey_ins, label_outs)
    
    
    rnd_img_outs = np.zeros((img_ins.shape[0], img_ins.shape[1], img_ins.shape[2], 3), dtype=np.uint8)
    for i in range(n_supix):
        rnd_img_outs[label_outs == i] = colors[i]
    
       
    label_out_folder = join(outfolder, 'label')
    rnd_out_folder = join(outfolder, 'rnd')
    avg_out_folder = join(outfolder, 'avg')
    avg_grey_out_folder = join(outfolder, 'avg_grey')
    

    for folder in [outfolder, label_out_folder, rnd_out_folder, avg_grey_out_folder, avg_out_folder]:
        if not exists(folder):
            makedirs(folder)

    for i, f in enumerate(onlyfiles):

        skimage.io.imsave(join(label_out_folder, f[:-3]+'png'), label_outs[:, :, i])
        skimage.io.imsave(join(rnd_out_folder, f[:-3]+'png'), rnd_img_outs[:, :, i, :])
        skimage.io.imsave(join(avg_out_folder, f[:-3]+'png'), avg_outs[:, :, i, :])
        skimage.io.imsave(join(avg_grey_out_folder, f[:-3]+'png'), avg_grey_outs[:, :, i])

    if save_vid:
        w, h, _, _ = img_ins.shape
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out_video = join(outfolder, '{}_3d_{}.mp4'.format(onlyfiles, n_supix))
        vid_out = cv2.VideoWriter(out_video,fourcc, 20.0, (h*2, w*2))

    
        for i, f in enumerate(onlyfiles):
            frame_rnd = np.concatenate((img_ins[:, :, i, :], rnd_img_outs[:, :, i, :]), axis=1)

            grey_3_channel = skimage.color.gray2rgb(avg_grey_outs[:, :, i])
            frame_avg = np.concatenate((avg_outs[:, :, i, :], grey_3_channel), axis=1)

            frame = np.concatenate((frame_rnd,frame_avg))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid_out.write(frame)
        
        vid_out.release()



def add_params(config, scene_id, params):
    
    if config.OF_EDGE_PREFACTORS:
        if config.OF_TOLERANCE_SQES:
            
            param = Boruvka3DOFReverseParam(config, scene_id, params)
        else:
            param = Boruvka3DOFParam(config, scene_id, params)        
    else:
        param = Boruvka3DParam(config, scene_id, params) 

def process_folders():
    
    if not exists(config.OUT_ROOT_FOLDER):
            makedirs(config.OUT_ROOT_FOLDER)
    params = []
    scene_ids = listdir(config.ROOT_FOLDER)
    scene_ids.sort()
    for scene_id in scene_ids:        
        add_params(config, scene_id, params)

    pool = Pool(processes=config.PROCESS_NUMBER) 
    pool.map(boruvkasupix3D, params)


def main():    
    process_folders()

if __name__ == '__main__':
    sys.exit(main())

