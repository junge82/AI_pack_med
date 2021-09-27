#!/bin/bash
###superpixel###

python3 train.py config15/train_wt_ax.txt

python3 util/rename_variables.py config15/test_wt_class_few.txt

python3 train.py config15/train_wt_cr.txt
python3 train.py config15/train_wt_sg.txt

python3 test.py config15/test_wt_class_few.txt

python3 util/evaluation.py

#python3 test.py config15/test_wt_class.txt