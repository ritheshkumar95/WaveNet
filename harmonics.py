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

# Hyperparams
NB_EPOCH=100
BATCH_SIZE = 8
FRAME_SIZE = 0 # How many samples per frame
Q_LEVELS = None # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
#DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
DATA_PATH='/home/rithesh/DeepLearning/Vocal Synthesis/data'
N_FILES = 8
BITRATE = 16000

Q_ZERO = None # Discrete value correponding to zero amplitude
N_BLOCKS=1
RF=N_BLOCKS*1024-N_BLOCKS+2
SEQ_LEN=2*RF
n_filters=256
#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def network(input_sequences,start_t):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    inp = input_sequences[:,None,None,:]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
    conv1 = lib.ops.conv1d("causal-conv",inp,2,1,n_filters,1,bias=False,batchnorm=False,pad=(0,1))[:,:,:,:length]
    prev_conv = conv1
    #prev_skip = []
    prev_skip = T.zeros_like(conv1)
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetConv1d("Block-%d"%i,prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False,dilation=value)
        prev_conv = x
        prev_skip += y
    out = T.nnet.relu(prev_skip)
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,n_filters,bias=False,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=False,batchnorm=False)

    amps = output[:,:,0,RF-1:].transpose(0,2,1)
    ang = T.arange(start_t,start_t+amps.shape[-2]).reshape((amps.shape[-2],1))*T.arange(1,129)*2*np.pi
    sig = amps[:,:,:128]*T.cos(ang) + amps[:,:,128:]*T.sin(ang)
    return sig.sum(axis=-1)

def network_gen(input_sequences,start_t):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    inp = input_sequences[:,None,None,:]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
    conv1 = lib.ops.conv1d("causal-conv",inp,2,1,n_filters,1,bias=False,batchnorm=False)
    prev_conv = conv1
    #prev_skip = []
    prev_skip = T.zeros_like(conv1[:,:,:,0])
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetGenConv1d("Block-%d"%i,prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False)
        prev_conv = x
        prev_skip += y[:,:,:,-1]
    out = T.nnet.relu(prev_skip[:,:,:,None])
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,n_filters,bias=False,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=False,batchnorm=False)
    amps = output[:,:,0,-1]
    ang = start_t *T.arange(1,129)*2*np.pi
    sig = amps[:,:128]*T.cos(ang) + amps[:,128:]*T.sin(ang)
    return sig.sum(axis=-1)


print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

sequences   = T.fmatrix('sequences')
input_sequences = sequences[:,:-1]
target_sequences = sequences[:,RF:]
start_t = T.iscalar()

predicted_sequences = network(input_sequences,start_t)
cost = T.sqr(predicted_sequences-target_sequences).mean()
#lib.load_params('iter_latest_wavenet.p')
# cost = T.nnet.categorical_crossentropy(
#     predicted_sequences,
#     target_sequences.flatten()
# ).mean()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
#cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.print_params_info(cost, params)
#updates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)
grads = T.grad(cost, wrt=params)
lr = T.fscalar()
updates = lasagne.updates.adam(grads, params, learning_rate=lr)

print "Gradients Computed"

train_fn = theano.function(
    [sequences,start_t,lr],
    [cost,predicted_sequences],
    updates=updates,
    on_unused_input='warn'
)


print "Training!"
DATA_PATH="/data/lisatmp3/kumarrit/blizzard"
for epoch in xrange(NB_EPOCH):
    costs = []
    times = []
    #data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO,RF))
    data_feeder = list(dataset.preprocess(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN,RF))
    results = []
    print "Epoch : ",epoch
    total_iters = 0
    i=0
    for seqs, t, m, s  in data_feeder:
        start_time = time.time()
        cost,pred = train_fn(seqs,t+RF,0.001)
        results.append(pred)
        i += 1
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters," (Epoch %d)"%(epoch)
        costs.append(cost)
        print "\tCost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
    del results


def plot(i):
    import matplotlib.pyplot as plt
    f,axarr = plt.subplots(8)
    for j in xrange(8):
        axarr[j].plot(data[i][0][j][1025:])
        axarr[j].plot(results[i][j],color='green')
    plt.show()
