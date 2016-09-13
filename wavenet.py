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
BATCH_SIZE = 32
FRAME_SIZE = 1024 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
#DATA_PATH = '/home/rithesh/DeepLearning/Vocal Synthesis/data'
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 50
SEQ_LEN = 1024 # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

def network(input_sequences):
    #inp = input_sequences[:,None,None,:]
    #length = inp.shape[-1]
    #Embedding layer seems to work well than 1-d convolution at the first layer
    emb = lib.ops.Embedding(
        'Embedding',
        Q_LEVELS,
        Q_LEVELS,
        input_sequences,
    ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    # Uncomment below to use 1x1 convolutions for the first layer
    # conv1 = lib.ops.conv1d("causal-conv",inp.astype('float32'),1,1,256,1,bias=True,batchnorm=True)[:,:,:,:length]

    #Filter Dilations used by DeepMind, Use this to tweak the no. of network blocks
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*1]).tolist()[0]

        ########################################################
        #Uncomment this to tweak the block internally
        ########################################################
        # conv1,skip1 = lib.ops.WaveNetConv1d("Block-1",emb,2,128,256,bias=True,batchnorm=True,dilation=1)
        # conv2,skip2 = lib.ops.WaveNetConv1d("Block-2",conv1,2,128,256,bias=True,batchnorm=True,dilation=2)
        # conv3,skip3 = lib.ops.WaveNetConv1d("Block-3",conv2,2,128,256,bias=True,batchnorm=True,dilation=4)
        # conv4,skip4 = lib.ops.WaveNetConv1d("Block-4",conv3,2,512,256,bias=True,batchnorm=True,dilation=8)
        # conv5,skip5 = lib.ops.WaveNetConv1d("Block-5",conv4,2,512,256,bias=True,batchnorm=True,dilation=16)
        # conv6,skip6 = lib.ops.WaveNetConv1d("Block-6",conv5,2,512,256,bias=True,batchnorm=True,dilation=32)
        # conv7,skip7 = lib.ops.WaveNetConv1d("Block-7",conv6,2,1024,256,bias=True,batchnorm=True,dilation=64)
        # conv8,skip8 = lib.ops.WaveNetConv1d("Block-8",conv7,2,1024,256,bias=True,batchnorm=True,dilation=128)
        # conv9,skip9 = lib.ops.WaveNetConv1d("Block-9",conv8,2,1024,256,bias=True,batchnorm=True,dilation=256)
        # conv10,skip10 = lib.ops.WaveNetConv1d("Block-10",conv9,2,256,256,bias=True,batchnorm=True,dilation=512)

    prev_conv = emb
    # prev_skip accumulates the skip connections from each residual block
    prev_skip = T.zeros_like(emb)
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetConv1d("Block-%d"%i,prev_conv,2,256,256,bias=True,batchnorm=True,dilation=value)
        prev_conv = x
        #accumulate skip connections
        prev_skip += y

    out = T.nnet.relu(prev_skip)
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,256,bias=True,batchnorm=True))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=True,batchnorm=True)
    #Receptive field for 30 resnet blocks is 3072. For 10 Resnet blocks with (1,2,4,8..512) dilations is 1024.
    #1023rd sample should predict 1024, etc.
    return output[:,:,0,1023:-1].transpose(0,2,1).reshape((-1,Q_LEVELS))


print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

sequences   = T.imatrix('sequences')
input_sequences = sequences[:,:]
target_sequences = sequences[:,1024:]

predicted_sequences = network(input_sequences)
cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(predicted_sequences.reshape((-1,Q_LEVELS))),
    target_sequences.flatten()
).mean()


params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.print_params_info(cost, params)
grads = T.grad(cost, wrt=params)
lr = T.fscalar()
updates = lasagne.updates.adam(grads, params, learning_rate=lr)

print "Gradients Computed"

train_fn = theano.function(
    [sequences,lr],
    [cost,T.nnet.softmax(predicted_sequences)],
    updates=updates,
    on_unused_input='warn'
)


print "Training!"
total_iters = 0

NB_EPOCH=100
for epoch in xrange(NB_EPOCH):
    costs = []
    times = []
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO)

    for seqs, reset in data_feeder:
        start_time = time.time()
        cost,preds = train_fn(seqs,0.001)
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters
        costs.append(cost)
        print "\tCost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
