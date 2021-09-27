import sys, getopt, math, os
import numpy as np
from time import time
from keras import callbacks
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import optimizers
from keras.utils import Sequence, multi_gpu_model,to_categorical
from keras.layers import Activation

from niftynet.layer.loss_segmentation import LossFunction

from util.data_generator import *
from util.parse_config import parse_config
from util.MSNet import MSNet


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()


def traink(config_file):

    gpu = 1
        
    config = parse_config(config_file)
    print(config)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type  = config_net['net_type']
    model = NetFactory.create(net_type)

    class_num   = config_net['class_num']
    loss = LossFunction(n_class=class_num)
    
    train_generator = DataGenerator(config_data)
    
    val_generator = DataGenerator(config_data)
    

    opt = optimizers.Adam()

    if gpu > 1:
        model = multi_gpu_model(model, gpus=gpu)
    model.compile(loss=loss, optimizer=opt)

    log_dir=os.path.join("logs/brats17_{}/".format(time()))
    print('[INFO] log_dir: ', log_dir)

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    remoteMonitor = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=6, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    modelCheckpoint_path = os.path.join(log_dir, 'nn_cluster.hrmdf5')
    modelCheckpoint = callbacks.ModelCheckpoint(modelCheckpoint_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    #if weightsPath is None:
    print("[INFO] training...")
    max_epoch = config_train['maximal_iteration']
    model.fit_generator(train_generator, 
        epochs=max_epoch,
        verbose=1,
        callbacks=[ earlyStopping,
                    remoteMonitor,
                    modelCheckpoint],
        max_queue_size=10,
        validation_data=val_generator,
        workers=4, 
        use_multiprocessing=True, 
        shuffle=True, 
        initial_epoch=0)

    print("[INFO] evaluating...")
    model.load_weights(modelCheckpoint_path)


    #train_pred = model.predict(X_train)
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    traink(config_file)
