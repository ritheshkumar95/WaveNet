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
NB_EPOCH=50
BATCH_SIZE = 32
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 100
BITRATE = 16000
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
N_BLOCKS=1
RF=N_BLOCKS*1024-N_BLOCKS+2
FRAME_SIZE = RF # How many samples per frame
SEQ_LEN=FRAME_SIZE+RF
n_filters=512

def network_gen(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
    # start = lib.ops.Embedding(
    #     'Embedding',
    #     Q_LEVELS,
    #     Q_LEVELS,
    #     input_sequences,
    # ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,256,bias=False,batchnorm=False)
    prev_conv = conv1
    prev_skip = []
    #prev_skip = T.zeros_like(conv1[:,:,:,0])
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetGenConv1d("Block-%d"%i,prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False)
        prev_conv = x
        #prev_skip += y[:,:,:,-1]
        prev_skip += [y[:,:,:,-1]]
    #out = T.nnet.relu(prev_skip[:,:,:,None])
    out = T.nnet.relu(T.sum(prev_skip,axis=0))[:,:,:,None]
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,n_filters,bias=False,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=False,batchnorm=False)
    return output[:,:,0,0]


tag='test_iter'
def write_audio_file(name, data):

    data = data.astype('float32')

    data -= data.min()
    data /= data.max()
    data -= 0.5
    data *= 0.95

    def invulaw(y,u=255):
        y = np.sign(y)*(1./u)*(np.power(1+u,np.abs(y))-1)
        return y

    import scipy.io.wavfile
    scipy.io.wavfile.write(name+'.wav',BITRATE,invulaw(data))
    #scipy.io.wavfile.write(name+'.wav',BITRATE,data)

test_sequences = T.imatrix()
start_t = T.iscalar()
output = network_gen(test_sequences)
test_fn = theano.function(
    [test_sequences],
    output
)
print "Compiled!"

def softmax(x, temp):
    x = x / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(prob_arr,sample_temperature):
    out = []
    import numpy.random as _rnd
    for i in xrange(prob_arr.shape[0]):
        output_dist = softmax(prob_arr[i], sample_temperature)
        output_dist = output_dist / np.sum(output_dist + 1e-7)
        out  += [_rnd.multinomial(1, output_dist).tolist().index(1)]
    return np.asarray(out,dtype=np.uint8)

# Generate 5 sample files, each 5 seconds long
N_SEQS = 8
LENGTH = 8*BITRATE

data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))
print "File loaded"
data = data_feeder[50][0][:]
samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
samples[:, :RF] = data[:N_SEQS,:RF]

for t in xrange(RF, LENGTH):
    probs = test_fn(samples[:,t-RF:t])
    samples[:,t] = sample(probs,0.001)
    #samples[:,t] = probs.flatten()
    print t, samples[:,t]

for i in xrange(N_SEQS):
    write_audio_file("sample_{}_{}".format(tag, i), samples[i][RF:])
