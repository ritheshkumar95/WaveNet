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
NB_EPOCH=10
BATCH_SIZE = 8
FRAME_SIZE = 0 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 8
BITRATE = 16000
SEQ_LEN = 1024 # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
RF=1024

tag='test_iter'
def write_audio_file(name, data):

    data = data.astype('float32')
    data -= data.min()
    data /= data.max()
    data -= 0.5
    data *= 0.95

    import scipy.io.wavfile
    scipy.io.wavfile.write(name+'.wav',BITRATE,data)

test_sequences = T.imatrix()
output = network(test_sequences)
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
data = data_feeder[5][0][:]
samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
samples[:, :SEQ_LEN] = data[:,:SEQ_LEN]

for t in xrange(SEQ_LEN, LENGTH):
    probs = test_fn(samples[:,t-1024:t])
    samples[:,t] = sample(probs,1)
    print t, samples[:,t]

for i in xrange(N_SEQS):
    write_audio_file("sample_{}_{}".format(tag, i), samples[i])
