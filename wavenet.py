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
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 8
BITRATE = 16000

Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
N_BLOCKS=1
RF=N_BLOCKS*1024-N_BLOCKS+2
SEQ_LEN=2*RF
n_filters=256
#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def network(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    #inp = input_sequences[:,None,None,:]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
#    start = lib.ops.Embedding(
#        'Embedding',
#        Q_LEVELS,
#        Q_LEVELS,
#        input_sequences,
#    ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,256,bias=False,batchnorm=False,pad=(0,1))[:,:,:,:length]
    # conv1,skip1 = lib.ops.WaveNetConv1d("Block-1",emb,2,128,256,bias=True,batchnorm=True,dilation=1)
    # conv2,skip2 = lib.ops.WaveNetConv1d("Block-2",conv1,2,128,256,bias=True,batchnorm=True,dilation=2)
    # conv3,skip3 = lib.ops.WaveNetConv1d("Block-3",conv2,2,128,256,bias=True,batchnorm=True,dilation=4)
    # conv4,skip4 = lib.ops.WaveNetConv1d("Block-4",conv3,2,512,256,bias=True,batchnorm=True,dilation=8)
    # conv5,skip5 = lib.ops.WaveNetConv1d("Block-5",conv4,2,512,256,bias=True,batchnorm=True,dilation=16)
    # conv6,skip6 = lib.ops.WaveNetConv1d("Block-6",conv5,2,512,256,bias=True,batchnorm=True,dilation=32)
    # conv7,skip7 = lib.ops.WaveNetConv1d("Block-7",conv6,2,1024,256,bias=True,batchnorm=True,dilation=64)
    # conv8,skip8 = lib.ops.WaveNetConv1d("Block-8",conv7,2,1024,256,bias=True,batchnorm=True,dilation=128)
    # conv9,skip9 = lib.ops.WaveNetConv1d("Block-9",conv8,2,1024,256,bias=True,batchnorm=True,dilation=256)
    # conv10,skip10 = lib.ops.WaveNetConv1d("Block-10",conv9,2,1024,512,bias=True,batchnorm=True,dilation=512)
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

sequences   = T.imatrix('sequences')
input_sequences = sequences[:,:-1]
target_sequences = sequences[:,RF:]

predicted_sequences = T.nnet.softmax(network(input_sequences))
#lib.load_params('iter_latest_wavenet.p')
cost = T.nnet.categorical_crossentropy(
    predicted_sequences,
    target_sequences.flatten()
).mean()

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
    [cost,predicted_sequences],
    updates=updates,
    on_unused_input='warn'
)


print "Training!"
for epoch in xrange(NB_EPOCH):
    costs = []
    times = []
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO,RF)
    print "Epoch : ",epoch
    total_iters = 0
    for seqs, reset in data_feeder:
        start_time = time.time()
        cost,pred = train_fn(seqs,0.001)
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters
        costs.append(cost)
        print "\tCost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
