import os, sys
sys.setrecursionlimit(10000)
import numpy as np
import numpy
numpy.random.seed(123)
import random
random.seed(123)
import dataset
import theano
import theano.tensor as T
theano.config.floatX='float32'
import lib.ops
import scipy.io.wavfile
import time
import lasagne
from six import iteritems
from platoon.channel import Worker
from platoon.param_sync import EASGD

worker = Worker(control_port=5567)
SEED = 123
lib.random_seed = SEED+worker.global_rank
# Hyperparams
NB_EPOCH=100
BATCH_SIZE = 8
FRAME_SIZE = 0 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 1000
BITRATE = 16000

Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
N_BLOCKS=1
RF=N_BLOCKS*1024-N_BLOCKS+2
SEQ_LEN=2*RF
n_filters=128
#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def floatX(arr):
    return np.asarray(arr, dtype=theano.config.floatX)

def adam(lr, tparams, grads, sequences, cost, epsilon=1e-8,beta1=0.9,beta2=0.999):

    zipped_grads = [lib.param('%s_grad' % k,p.get_value() * floatX(0.))
                    for k, p in iteritems(tparams)]
    running_grads = [lib.param('%s_rgrad' % k,p.get_value() * floatX(0.))
                     for k, p in iteritems(tparams)]
    running_grads2 = [lib.param('%s_rgrad2' % k,p.get_value() * floatX(0.))
                      for k, p in iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, beta1 * rg + (1-beta1) * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, beta2 * rg2 + (1-beta2) * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    t_prev = lib.param('t_prev',floatX(0.))
    one = T.constant(1)
    t = t_prev+1
    a_t = lr*T.sqrt(1-beta2**t)/(1-beta1**t)

    f_grad_shared = theano.function([sequences], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='adam_f_grad_shared')

    updir = [lib.param('%s_updir' % k,p.get_value() * floatX(0.))
             for k, p in iteritems(tparams)]

    updir_new = [(ud, a_t * rg / T.sqrt(rg2 + epsilon))
                 for ud, rg, rg2 in zip(updir, running_grads, running_grads2)]
    param_up = [(p, p - udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up + [(t_prev,t)] ,
                               on_unused_input='ignore',
                               name='adam_f_update')

    return f_grad_shared, f_update

def network(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    #inp = input_sequences[:,None,None,:]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
    start = lib.ops.Embedding(
       'Embedding',
       Q_LEVELS,
       Q_LEVELS,
       input_sequences,
    ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    #start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,256,bias=False,batchnorm=False,pad=(0,1))[:,:,:,:length]
    prev_conv = conv1
    #prev_skip = []
    prev_skip = T.zeros_like(conv1)
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetConv1d("Block-%d"%i,prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False,dilation=value)
        prev_conv = x
        prev_skip += y
    #    prev_skip += [y]

    #out = T.nnet.relu(T.sum(prev_skip,axis=0))
    out = T.nnet.relu(prev_skip)
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,n_filters,bias=False,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=False,batchnorm=False)
    return output[:,:,0,RF-1:].transpose(0,2,1).reshape((-1,Q_LEVELS))


print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def build_model(train_len=10):
    sequences   = T.imatrix('sequences')
    input_sequences = sequences[:,:-1]
    target_sequences = sequences[:,RF:]

    predicted_sequences = T.nnet.softmax(network(input_sequences))
    #lib.load_params('iter_latest_wavenet.p')
    cost = T.nnet.categorical_crossentropy(
        predicted_sequences,
        target_sequences.flatten()
    ).mean()

    params = lib.search(cost, lambda x: hasattr(x, 'param'))
    tparams = {p.name:p for p in params}
    lib.print_params_info(cost, params)
    #updates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)
    grads = T.grad(cost, wrt=list(tparams.values()))
    lr = T.fscalar()

    f_grad_shared,f_update = adam(lr,tparams,grads,sequences,cost)

    list_tparams = list(tparams.values())
    worker.init_shared_params(list_tparams, param_sync_rule=EASGD(0.5))

    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO,RF)

    worker.copy_to_local()
    while True:
        step = worker.send_req('next')
        print(step)

        if step == 'train':
            use_noise.set_value(numpy_floatX(1.))
            for i in range(train_len):
                seqs,reset = next(data_feeder)
                cost = f_grad_shared(seqs)
                f_update(0.001)
            print('Train cost:', cost)
            step = worker.send_req('done', {'train_len': train_len})

            print("Syncing with global params")
            worker.sync_params(synchronous=True)

        if step == 'stop':
            break

    # Release all shared resources.
    worker.close()

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--valid_sync', dest='valid_sync', action='store_true', default=False)
    parser.add_argument('--param-sync-api', action='store_true', default=False)
    args = parser.parse_args()

    build_model(train_len=100)
