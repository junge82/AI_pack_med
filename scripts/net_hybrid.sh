#!/bin/bash
# reset the initial maps
python3 net_download.py hybrid_mr_ct_model_zoo -r
#python3 net_run.py train \
#    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
#    -c /home/junge82/niftynet/extensions/isampler_autocontext_mr_ct/net_hybrid.ini \
#    --starting_iter 0 --max_iter 500
for max_iter in `seq 1000 1000 10000`
do
    python3 net_run.py inference \
        -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
        -c ~/niftynet/extensions/isampler_autocontext_mr_ct/net_hybrid.ini \
        --inference_iter -1 --spatial_window_size 240,240,1 --batch_size 4 \
        --dataset_split_file nofile  --error_map True
        
    python3 net_run.py inference \
        -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
        -c ~/niftynet/extensions/isampler_autocontext_mr_ct/net_hybrid.ini \
        --inference_iter -1 --spatial_window_size 240,240,1 --batch_size 4 --dataset_split_file nofile

    python3 net_run.py train \
        -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
        -c ~/niftynet/extensions/isampler_autocontext_mr_ct/net_hybrid.ini \
        --starting_iter -1 --max_iter $max_iter
done
