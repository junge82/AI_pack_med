#!/bin/bash
###superpixel###

#python3 train.py config15/train_wt_ax_sp.txt
#python3 train.py config15/train_tc_ax_sp.txt
#python3 train.py config15/train_en_ax_sp.txt

#python3 util/rename_variables.py

#python3 train.py config15/train_wt_cr_sp.txt
#python3 train.py config15/train_wt_sg_sp.txt

python3 train.py config15/train_tc_cr_sp.txt
python3 train.py config15/train_tc_sg_sp.txt

python3 train.py config15/train_en_cr_sp.txt
python3 train.py config15/train_en_sg_sp.txt

#python3 test.py config15/test_all_class_sp_few.txt

#python3 util/evaluation.py

#python3 test.py config15/test_wt_class.txt