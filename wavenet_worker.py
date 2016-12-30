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
import argparse
import pickle
import new_dataset
from model import network

worker = None
# Hyperparams
NB_EPOCH=200
BATCH_SIZE = 8
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 8192
BITRATE = 16000
GRAD_CLIP=1
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
Q_TYPE='linear'

N_BLOCKS=5
DILATION_DEPTH=10
RF=N_BLOCKS*(2**(DILATION_DEPTH))-N_BLOCKS+2
n_filters=64

#FRAME_SIZE = RF # How many samples per frame
#SEQ_LEN=2*RF
OVERLAP=RF
SEQ_LEN=1600

N_GPUS=4
alpha = 1./N_GPUS
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

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def build_model(worker,train_len=100,param_sync_api=True):
    sequences   = T.imatrix('sequences')
    input_sequences = sequences[:,:-1]
    target_sequences = sequences[:,RF:]

    # def network_my(input_sequences):
    #     batch_size = input_sequences.shape[0]
    #     length = input_sequences.shape[1]
    #     dilations = np.asarray([[2**i for i in xrange(DILATION_DEPTH)]*N_BLOCKS]).tolist()[0]
    #     #skip_weights = lib.param("scaling_weights", numpy.ones(len(dilations)).astype('float32'))
    #
    #     #start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    #     start =  (input_sequences.astype('float32')/lib.floatX(Q_LEVELS-1) - lib.floatX(0.5))[:,None,:,None]
    #     conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,1,bias=True,batchnorm=False,pad=(1,0))[:,:,:length,:]
    #     prev_conv = conv1
    #     #prev_skip = []
    #     prev_skip = T.zeros((batch_size,n_filters,length,1))
    #     for i,value in enumerate(dilations):
    #         prev_conv,y = lib.ops.WaveNetConv1d("Block-%d"%(i+1),prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False,dilation=value)
    #         #prev_skip += y*skip_weights[i]
    #         prev_skip += y
    #         #prev_skip += [y]
    #
    #     #out = T.nnet.relu(T.sum(prev_skip,axis=0))
    #     out = T.nnet.relu(prev_skip)
    #     #out = prev_skip
    #     out = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))
    #     out = T.nnet.relu(lib.ops.conv1d("Output.2",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))
    #     out = T.nnet.relu(lib.ops.conv1d("Output.3",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))
    #
    #     out = lib.ops.conv1d("Output.4",out,1,1,256,n_filters,bias=True,batchnorm=False)
    #
    #     return out[:,:,RF-1:,0].transpose(0,2,1).reshape((-1,Q_LEVELS))

    predicted_sequences = T.nnet.softmax(network(input_sequences))
    #lib.load_params('iter_latest_wavenet.p')
    cost = T.nnet.categorical_crossentropy(
        predicted_sequences,
        target_sequences.flatten()
    ).mean()

    cost = cost * lib.floatX(1.44269504089)


    params = lib.search(cost, lambda x: hasattr(x, 'param'))
    tparams = {p.name:p for p in params}

    copy_params = lambda tparams: {x:theano.shared(y.get_value(),name=x) for x,y in tparams.iteritems()}

    lib.print_params_info(cost, params)
    #updates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)

    list_tparams = list(tparams.values())

    if param_sync_api:
        worker.init_shared_params(list_tparams, param_sync_rule=EASGD(alpha))
    else:
        from platoon.training import global_dynamics as gd
        cparams = copy_params(tparams)
        list_cparams = list(cparams.values())
        easgd = gd.EASGD(worker)
        easgd.make_rule(list_tparams, list_cparams, alpha)


    grads = T.grad(cost, wrt=list_tparams)
    grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

    lr = T.fscalar()

    f_grad_shared,f_update = adam(lr,tparams,grads,sequences,cost)

    def save_params(_params,path):
        param_vals = {}
        for name, param in _params.iteritems():
            param_vals[name] = param.get_value()

        with open(path, 'wb') as f:
            pickle.dump(param_vals, f)

    if param_sync_api:
        worker.copy_to_local()

    costs = []
    #data_feeder = dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES, True,worker.global_rank())
    new_dataset.random_seed = worker._worker_id
    data_feeder = new_dataset.blizz_train_feed_epoch(BATCH_SIZE,SEQ_LEN,OVERLAP,Q_LEVELS,Q_ZERO,Q_TYPE)

    iter_count=0
    while True:
        step = worker.send_req('next')

        if step == 'train':
            for i in range(train_len):
                try:
                    seqs,reset,mask = next(data_feeder)
                except StopIteration:
                    #data_feeder = dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES, True,worker.global_rank())
                    new_dataset.random_seed = worker._worker_id
                    data_feeder = new_dataset.blizz_train_feed_epoch(BATCH_SIZE,SEQ_LEN,OVERLAP,Q_LEVELS,Q_ZERO,Q_TYPE)
                    seqs,reset,mask = next(data_feeder)
                    print('Train cost:', np.mean(costs))
                    costs = []

                costs.append(f_grad_shared(seqs))
                f_update(0.001)
                iter_count += 1

            step = worker.send_req('done', {'train_len': train_len})
            if iter_count%5000==0:
                print('Train cost:',np.mean(costs))

            if param_sync_api:
                #print("Syncing with global params")
                worker.sync_params(synchronous=True)
            else:
                easgd()

        if step=='save':
            if param_sync_api:
                save_params(tparams,"worker_%d.p"%worker._worker_id)
                step = worker.send_req('saved')
                print('Saving now')
            else:
                save_params(cparams,"worker%d.p"%worker._worker_id))
                step = worker.send_req('saved')

        if step == 'stop':
            break

    # Release all shared resources.
    worker.close()

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    global worker
    parser = Worker.default_parser()
    parser.add_argument('--valid_sync', dest='valid_sync', action='store_true', default=False)
    parser.add_argument('--param-sync-api', action='store_true', default=True)
    #SEED = 123
    #lib.random_seed = SEED+worker._worker_id
    args = parser.parse_args()
    worker = Worker(**Worker.default_arguments(args))

    build_model(worker,train_len=10,param_sync_api=args.param_sync_api)
