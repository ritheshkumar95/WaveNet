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
import vctk_dataset
import tqdm
from tqdm import tqdm
import new_dataset

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

#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def network(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    dilations = np.asarray([[2**i for i in xrange(DILATION_DEPTH)]*N_BLOCKS]).tolist()[0]
    #skip_weights = lib.param("scaling_weights", numpy.ones(len(dilations)).astype('float32'))

    #start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    start =  (input_sequences.astype('float32')/lib.floatX(Q_LEVELS-1) - lib.floatX(0.5))[:,None,:,None]
    conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,1,bias=True,batchnorm=False,pad=(1,0))[:,:,:length,:]
    prev_conv = conv1
    #prev_skip = []
    prev_skip = T.zeros((batch_size,n_filters,length,1))
    for i,value in enumerate(dilations):
        prev_conv,y = lib.ops.WaveNetConv1d("Block-%d"%(i+1),prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False,dilation=value)
        #prev_skip += y*skip_weights[i]
        prev_skip += y
        #prev_skip += [y]

    #out = T.nnet.relu(T.sum(prev_skip,axis=0))
    out = T.nnet.relu(prev_skip)
    #out = prev_skip
    out = T.nnet.relu(lib.ops.conv1d("Output.1",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))
    out = T.nnet.relu(lib.ops.conv1d("Output.2",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))
    out = T.nnet.relu(lib.ops.conv1d("Output.3",out,1,1,n_filters,n_filters,bias=True,batchnorm=False))

    out = lib.ops.conv1d("Output.4",out,1,1,256,n_filters,bias=True,batchnorm=False)

    return out[:,:,RF-1:,0].transpose(0,2,1).reshape((-1,Q_LEVELS))

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
cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.print_params_info(cost, params)
#updates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)
grads = T.grad(cost, wrt=params)
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

lr = T.fscalar()
updates = lasagne.updates.adam(grads, params, learning_rate=lr)

print "Gradients Computed"

train_fn = theano.function(
    [sequences,lr],
    [cost],
    updates=updates,
    on_unused_input='warn'
)

print "Compiled Train Function"

test_fn = theano.function(
    [sequences],
    [cost],
    on_unused_input='warn'
)

print "Compiled Test Function"

generate_fn = theano.function(
    [sequences],
    [lib.ops.softmax_and_sample(network(sequences))],
    on_unused_input='warn'
)

print "Compiled Generate Function"

def generate(generate_fn):
    tag = 'test_iter'
    N_SEQS = 8
    LENGTH = 3*BITRATE
    samples = numpy.full((N_SEQS, LENGTH), fill_value = Q_ZERO, dtype=np.uint8)

    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        import scipy.io.wavfile
        scipy.io.wavfile.write(name+'.wav',BITRATE,data)

    #data = data_feeder.next()
    #data_feeder = list(dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES, True, 34965))
    #data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))
    #data_feeder = list(vctk_dataset.feed_epoch(225, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES))
    print "File loaded"
    #data = data[0][:]
    #samples[:, :RF] = data[:N_SEQS,:RF]

    for t in xrange(RF, LENGTH):
        samples[:,t] = generate_fn(samples[:,t-RF:t])[0]
        #samples[:,t] = probs.flatten()
        print t, samples[:,t]

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i][RF:])

print "Training!"
for epoch in xrange(1,NB_EPOCH):
    costs = []
    times = []
    #data_feeder = dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES)
    #data_feeder = vctk_dataset.feed_epoch(225, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES)
    data_feeder = new_dataset.blizz_train_feed_epoch(BATCH_SIZE,SEQ_LEN,OVERLAP,Q_LEVELS,Q_ZERO,Q_TYPE)
    print "Epoch : ",epoch
    total_iters = 0
    for seqs,reset,mask in tqdm(data_feeder):
        total_iters += 1
        start_time = time.time()
        cost,pred = train_fn(seqs,0.001)
        total_time = time.time() - start_time
        costs.append(cost)
        times.append(total_time)
        if total_iters%1000==0:
            print "\tCost : ", np.mean(costs)
            print "\tTime : ", np.mean(times)

    print "\tCost : ", np.mean(costs)
    print "\tTime : ", np.mean(times)
    #if epoch%50==0:
    #    generate()
