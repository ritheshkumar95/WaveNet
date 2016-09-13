import os, sys
sys.path.append(os.getcwd())

import numpy as np
import numpy
numpy.random.seed(123)
import random
random.seed(123)

import dataset

import theano
import theano.tensor as T
theano.config.floatX='float32'
from theano.tensor.nnet import neighbours
import theano.ifelse
import lib
import lib.optimizers
import lasagne
import scipy.io.wavfile

import time
import functools
import itertools

# Hyperparams
NB_EPOCH=10
BATCH_SIZE = 128
N_FRAMES = 256 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 4 # How many samples per frame
DIM = 512 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
N_GRUS = 2 # How many GRUs to stack in the frame-level model
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 1000
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = N_FRAMES * FRAME_SIZE # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def frame_level_rnn(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, N_FRAMES * FRAME_SIZE)
    h0.shape:              (batch size, N_GRUS, DIM)
    reset.shape:           ()
    output.shape:          (batch size, N_FRAMES * FRAME_SIZE, DIM)
    """
    batch_size = input_sequences.shape[0]
    n_frames = input_sequences.shape[1]/FRAME_SIZE

    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_GRUS, DIM), dtype=theano.config.floatX)
    )

    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_GRUS, DIM)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] / (FRAME_SIZE * FRAME_SIZE),
        FRAME_SIZE*FRAME_SIZE
    ))

    # frames = emb.reshape((
    #     input_sequences.shape[0],
    #     input_sequences.shape[1] / (FRAME_SIZE*FRAME_SIZE),
    #     FRAME_SIZE*Q_LEVELS
    # ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    # frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    # frames *= lib.floatX(2)

    gru1 = lib.ops.myGRU('FrameLevel.GRU1', FRAME_SIZE*FRAME_SIZE, DIM, frames, h0=h0[:, 0])

    gru1_output = lib.ops.Dense(
        'FrameLevel1.Output',
        DIM,
        FRAME_SIZE * DIM,
        gru1,
        init='he',
        hidden_dim=64
    ).reshape((batch_size,256,DIM))


    gru2 = lib.ops.myGRU('FrameLevel.GRU2', DIM, DIM, gru1_output, h0=h0[:, 1])
    #gru3 = lib.ops.myGRU('FrameLevel.GRU3', DIM, DIM, gru2, h0=h0[:, 2])

    #gru1,gru2,gru3 = lib.ops.myGRU('FrameLevel.GRU', FRAME_SIZE, DIM, frames, h0=h0)

    # gru3.shape = (batch_size,N_FRAMES,DIM)

    gru2_output = lib.ops.Dense(
        'FrameLevel2.Output',
        DIM,
        FRAME_SIZE * DIM,
        gru2,
        init='he',
        hidden_dim=256
    ).reshape((batch_size,1024,DIM))


    last_hidden = T.stack([gru1[:, -1], gru2[:, -1]], axis=1)

    return (gru2_output, last_hidden)

def sample_level_predictor(frame_level_outputs, prev_samples):
    """
    frame_level_outputs.shape: (batch size*SEQ_LEN, DIM)
    prev_samples.shape:        (batch size*SEQ_LEN, FRAME_SIZE)
    output.shape:              (batch size*SEQ_LEN, Q_LEVELS)
    """

    prev_samples = lib.ops.Embedding(
        'SampleLevel.Embedding',
        Q_LEVELS,
        Q_LEVELS,
        prev_samples
    ).reshape((-1, FRAME_SIZE * Q_LEVELS))

    # prev_samples.shape = (batch_size*SEQ_LEN,FRAME_SIZE,Q_LEVELS)

    out = lib.ops.Dense(
        'SampleLevel.L1_PrevSamples',
        FRAME_SIZE * Q_LEVELS,
        DIM,
        prev_samples,
        bias=False,
        init='he',
    ) ##(128,256,512)
    out += frame_level_outputs
    out = T.nnet.relu(out)

    out = lib.ops.Dense('SampleLevel.L2', DIM, DIM, out, init='he')
    out = T.nnet.relu(out)

    out = lib.ops.Dense('SampleLevel.L3', DIM, DIM, out, init='he')
    out = T.nnet.relu(out)

    # We apply the softmax later
    return lib.ops.Dense('SampleLevel.Output', DIM, Q_LEVELS, out)

sequences   = T.imatrix('sequences')
h0          = T.tensor3('h0')
reset       = T.iscalar('reset')

input_sequences = sequences[:, :-FRAME_SIZE]
target_sequences = sequences[:, FRAME_SIZE:]

frame_level_outputs, new_h0 = frame_level_rnn(input_sequences, h0, reset)

# frame_level_outputs.shape = (batch_size,SEQ_LEN,DIM)

prev_samples = sequences[:, :-1]
prev_samples = prev_samples.reshape((1, BATCH_SIZE, 1, -1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1, FRAME_SIZE), neib_step=(1, 1), mode='valid')
prev_samples = prev_samples.reshape((BATCH_SIZE * SEQ_LEN, FRAME_SIZE))

sample_level_outputs = sample_level_predictor(
    frame_level_outputs.reshape((BATCH_SIZE * SEQ_LEN, DIM)),
    prev_samples
)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs),
    target_sequences.flatten()
).mean()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))

lib.print_params_info(cost, params)

#pdates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)
grads = T.grad(cost, wrt=params, disconnected_inputs='warn')

#grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

print "Gradients Computed"

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function(
    [sequences, h0, reset],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

frame_level_generate_fn = theano.function(
    [sequences, h0, reset],
    frame_level_rnn(sequences, h0, reset),
    on_unused_input='warn'
)

frame_level_outputs = T.matrix('frame_level_outputs')
prev_samples        = T.imatrix('prev_samples')
sample_level_generate_fn = theano.function(
    [frame_level_outputs, prev_samples],
    lib.ops.softmax_and_sample(
        sample_level_predictor(
            frame_level_outputs,
            prev_samples
        )
    ),
    on_unused_input='warn'
)

def generate_and_save_samples(tag):

    def write_audio_file(name, data):

        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95

        import scipy.io.wavfile
        scipy.io.wavfile.write(name+'.wav',BITRATE,data)

    # Generate 5 sample files, each 5 seconds long
    N_SEQS = 5
    LENGTH = 8*BITRATE

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :FRAME_SIZE] = Q_ZERO

    h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')
    frame_level_outputs = None

    for t in xrange(FRAME_SIZE, LENGTH):

        if t % FRAME_SIZE == 0:
            frame_level_outputs, h0 = frame_level_generate_fn(
                samples[:, t-FRAME_SIZE:t],
                h0,
                numpy.int32(t == FRAME_SIZE)
            )

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE],
            samples[:, t-FRAME_SIZE:t]
        )

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i])

print "Training!"
total_iters = 0

for epoch in xrange(NB_EPOCH):
    h0 = np.zeros((BATCH_SIZE, N_GRUS, DIM)).astype(theano.config.floatX)
    costs = []
    times = []
    data = dataset.get_data(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN+FRAME_SIZE, 0, Q_LEVELS, Q_ZERO)

    for seqs, reset in data:
        start_time = time.time()
        cost, h0 = train_fn(seqs, h0, reset)
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters
        costs.append(cost)
        print "\tCost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
        if total_iters%10000==0:
            generate_and_save_samples('iterno_%d'%total_iters)
    break
            # print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}\ttime per iter:{}".format(
            #     epoch,
            #     total_iters,
            #     numpy.mean(costs),
            #     total_time,
            #     total_time / total_iters
            # )
            # tag = "iters{}_time{}".format(total_iters, total_time)
            # generate_and_save_samples(tag)
            # lib.save_params('params_{}.pkl'.format(tag))

            # costs = []
            # last_print_time += PRINT_TIME
            # last_print_iters += PRINT_ITERS
