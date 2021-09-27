#!/usr/bin/env python3

# standard lib
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
    
import argparse
from readFlowFile import read

# numpy family
import numpy as np

import cv2


scene_folder = '/home/fothar/DAVIS/JPEGImages/480p/bear/'
out_scene_folder = '/home/fothar/boruvka_of_test/bear/'



onlyfiles = [f for f in listdir(scene_folder) if isfile(join(scene_folder, f))]
onlyfiles.sort()

y_center = 30
x_center = 40

#green = 128
q_w = 240 

for i, f in enumerate(onlyfiles):
    frame = cv2.imread(join(scene_folder, f))

    s = 20

    cv2.rectangle(frame,(0, 41),(400, 400),(255,255,255),-1)
    cv2.rectangle(frame,(0, 0),(400, 41),(0,0,0),-1)

    q = frame[y_center-s:y_center+s, x_center:x_center+q_w, :]
    q[:, :, 0] = 0

    for n in range(40):
        for m in range(q_w):
            print(n, m)
            q[n, m, 1] = 128 + 50 * np.sin(n/2.)*np.sin(m/2.)

    q[:, :, 2] = 0
    #cv2.rectangle(frame,(x_center, y_center-s),(x_center+140, y_center+s),(0,255,0),-1)
    
    #cv2.rectangle(frame,(40, y_center),(40+s, y_center+s),(255,255,255),-1)
    #cv2.rectangle(frame,(40-d, y_center-d),(40+s+d, y_center+s+d),(0,0,0),3)
    #cv2.rectangle(frame,(40, y_center),(40+s, y_center+s),(10,100,0),6)
    #cv2.rectangle(frame,(40, y_center),(40+s, y_center+s),(50,0,255),3)



    y_center += 90 if i == 15 else 3
    x_center += 1

    
    cv2.imwrite(join(out_scene_folder, f[:-3]+'png'), frame)




