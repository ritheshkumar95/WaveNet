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
import theano.tensor.fft

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
RF=N_BLOCKS*32-N_BLOCKS+2
SEQ_LEN=2*RF
n_filters=256
#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def network(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    inp = input_sequences[:,None,None,:]
    dilations = np.asarray([[1,2,4,8,16]*N_BLOCKS]).tolist()[0]
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
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,n_filters,n_filters,bias=False,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,34,n_filters,bias=False,batchnorm=False)

    result = output[:,:,0,-1]
    result2 = T.nnet.relu(lib.ops.Dense('Op.1',34,512,result,weightnorm=False))
    result3 = lib.ops.Dense('Op.2',512,34,result2,weightnorm=False)
    return output[:,:,0,-1].reshape((batch_size,17,2))

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

sequences   = T.fmatrix('sequences')
input_sequences = sequences[:,:RF]
target_sequences = sequences[:,RF:]

pred_freq = network(input_sequences)
target_freq = theano.tensor.fft.rfft(target_sequences)
cost = T.sqr(pred_freq-target_freq).mean()
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
    [sequences,lr],
    [cost,pred_freq],
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
        cost,pred = train_fn(seqs,0.001)
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
