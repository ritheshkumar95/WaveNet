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
N_FRAMES = 256 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 1024 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold
DIM = 512
# Dataset
#DATA_PATH = '/home/rithesh/DeepLearning/Vocal Synthesis/data'
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 50
BITRATE = 16000

TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = 1024 # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude


def network_gen(input_sequences):

#    inp = input_sequences[:,None,None,:]
#    length = inp.shape[-1]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*1]).tolist()[0]
    conv1 = lib.ops.Embedding(
        'Embedding',
        Q_LEVELS,
        Q_LEVELS,
        input_sequences,
    ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    prev_conv = conv1
    prev_skip = T.zeros_like(conv1)
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetConv1d("Block-%d"%i,prev_conv,2,256,256,bias=True,batchnorm=False,dilation=value)
        prev_conv = x
        prev_skip += y

    out = T.nnet.relu(prev_skip)
    out2 = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,256,256,bias=True,batchnorm=False))
    output = lib.ops.conv1d("Output.2",out2,1,1,256,256,bias=True,batchnorm=False)
    return output[:,:,0,-1].reshape((-1,Q_LEVELS))

tag='test_iter'
def write_audio_file(name, data):

    data = data.astype('float32')
    data -= data.min()
    data /= data.max()
    data -= 0.5
    data *= 0.95

    import scipy.io.wavfile
    scipy.io.wavfile.write(name+'.wav',BITRATE,data)

sequences = T.imatrix()
output = T.nnet.softmax(network_gen(sequences))
#lib.load_params('iter_2_wavenet.p')
test_fn = theano.function(
    [sequences],
    output
)
print "Compiled!"

# Generate 5 sample files, each 5 seconds long
N_SEQS = 4
LENGTH = 4*BITRATE
SEQ_LEN=1024

BATCH_SIZE = 4
FRAME_SIZE = 256 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 50
SEQ_LEN = 1024 # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

#Seed
data = data_feeder[10][0][:]
samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
samples[:, :SEQ_LEN] = data[:,:SEQ_LEN]

for t in xrange(SEQ_LEN, LENGTH):
    probs = test_fn(samples[:,t-1024:t])
    probs = probs.reshape((N_SEQS,Q_LEVELS))
    samples[:,t] = np.argmax(probs,axis=1)
    print t

for i in xrange(N_SEQS):
    write_audio_file("sample_{}_{}".format(tag, i), samples[i])
