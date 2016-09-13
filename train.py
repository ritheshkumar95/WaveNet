import os, sys
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
BATCH_SIZE = 32
N_FRAMES = 256 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 768 # How many samples per frame
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold
DIM = 512
# Dataset
DATA_PATH = '/home/rithesh/DeepLearning/Vocal Synthesis/data'
#DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 50
BITRATE = 16000

TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = 8192 # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

def network(input_sequences,h0,reset):

    batch_size = input_sequences.shape[0]

    learned_h0 = lib.param(
        'Session.h0',
        numpy.zeros(DIM, dtype=theano.config.floatX)
    )

    learned_h0 = T.alloc(learned_h0, h0.shape[0], DIM)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    emb = lib.ops.Embedding(
        'Embedding',
        Q_LEVELS,
        Q_LEVELS,
        input_sequences,
    ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)

    # conv1 = T.nnet.relu(lib.ops.conv1d("conv1",emb,3,1,128,256,bias=True,batchnorm=True)) #(32, 512, 1, 255)   - 289 - 31
    # conv2 = T.nnet.relu(lib.ops.conv1d("conv2",conv1,3,2,256,128,bias=True,batchnorm=True)) #(32, 512, 1, 127) - 143 - 15
    # conv3 = T.nnet.relu(lib.ops.conv1d("conv3",conv2,3,2,512,256,bias=True,batchnorm=True)) #(32, 512, 1, 63)  - 71  -  7
    # conv4 = T.nnet.relu(lib.ops.conv1d("conv4",conv3,3,2,1024,512,bias=True,batchnorm=True)) #(32, 512, 1, 31)  - 35  -  3
    # conv5 = T.nnet.relu(lib.ops.conv1d("conv5",conv4,3,2,2048,1024,bias=True,batchnorm=True)) #(32, 512, 1, 15) - 17  -  1
    start = lib.ops.ResNetConv1d("ResNet-Enc-0",emb,3,1,256,256,bias=True,batchnorm=True) # 8960 RF - 2
    rconv1 = lib.ops.ResNetConv1d("ResNet-Enc-1",start,3,2,128,256,bias=True,batchnorm=True) # 4480 RF - 5
    rconv2 = lib.ops.ResNetConv1d("ResNet-Enc-2",rconv1,3,2,128,128,bias=True,batchnorm=True) # 2240 RF - 11
    rconv3 = lib.ops.ResNetConv1d("ResNet-Enc-3",rconv2,3,2,128,128,bias=True,batchnorm=True) # 1120 RF - 23
    rconv4 = lib.ops.ResNetConv1d("ResNet-Enc-4",rconv3,3,2,256,128,bias=True,batchnorm=True) # 560 RF - 47
    rconv5 = lib.ops.ResNetConv1d("ResNet-Enc-5",rconv4,3,2,256,256,bias=True,batchnorm=True) # 280 RF - 95
    rconv6 = lib.ops.ResNetConv1d("ResNet-Enc-6",rconv5,3,2,256,256,bias=True,batchnorm=True) # 140 RF - 191
    rconv7 = lib.ops.ResNetConv1d("ResNet-Enc-7",rconv6,3,2,512,256,bias=True,batchnorm=True) #  70 RF - 383
    rconv8 = lib.ops.ResNetConv1d("ResNet-Enc-8",rconv7,3,2,512,512,bias=True,batchnorm=True) #  35 RF - 767

    #gru1 = lib.ops.myGRU('Encoder.GRU1',DIM,DIM,rconv7.transpose(2,0,3,1)[0][:,:15,:],h0=h0) # (32, 15, 512)
    gru1 = lib.ops.myGRU('Encoder.GRU1',DIM,DIM,rconv8.transpose(2,0,3,1)[0][:,:32,:],h0=h0) # (32, 15, 512)
    gru = gru1.transpose(0,2,1)[:,:,None,:] #(32, 512, 1, 15)
    #project = lib.ops.conv1d("Project.GRU",gru,1,1,4096,512,bias=True,batchnorm=True)

    rdeconv8 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-8",gru,3,2,512,512,bias=True,batchnorm=True)+rconv7[:,:,:,3:67]) # 64
    rdeconv7 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-7",rdeconv8,3,2,256,512,bias=True,batchnorm=True)+rconv6[:,:,:,9:137]) #128
    rdeconv6 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-6",rdeconv7,3,2,256,256,bias=True,batchnorm=True)+rconv5[:,:,:,21:277]) #256
    rdeconv5 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-5",rdeconv6,3,2,256,256,bias=True,batchnorm=True)+rconv4[:,:,:,45:557]) #512
    rdeconv4 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-4",rdeconv5,3,2,128,256,bias=True,batchnorm=True)+rconv3[:,:,:,93:1117]) #1024
    rdeconv3 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-3",rdeconv4,3,2,128,128,bias=True,batchnorm=True)+rconv2[:,:,:,189:2237]) #2048
    rdeconv2 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-2",rdeconv3,3,2,128,128,bias=True,batchnorm=True)+rconv1[:,:,:,381:4477]) #4096
    rdeconv1 = T.nnet.relu(lib.ops.ResNetDeconv1d("ResNet-Dec-1",rdeconv2,3,2,256,128,bias=True,batchnorm=True)+start[:,:,:,765:8957]) #8192
    rdeconv0 = lib.ops.ResNetDeconv1d("ResNet-Dec-0",rdeconv1,3,1,256,256,bias=True,batchnorm=True,act=False) #8192

    # deconv5 = T.nnet.relu(lib.ops.deconv1d("deconv5",gru,3,2,1024,2048,bias=True,batchnorm=True)+conv4[:,:,:,2:33]) # (32, 512, 1, 31)
    # deconv4 = T.nnet.relu(lib.ops.deconv1d("deconv4",deconv5,3,2,512,1024,bias=True,batchnorm=True)+conv3[:,:,:,6:69]) # (32, 512, 1, 63)
    # deconv3 = T.nnet.relu(lib.ops.deconv1d("deconv3",deconv4,3,2,256,512,bias=True,batchnorm=True)+conv2[:,:,:,14:141]) # (32, 512, 1, 127)
    # deconv2 = T.nnet.relu(lib.ops.deconv1d("deconv2",deconv3,3,2,128,256,bias=True,batchnorm=True)+conv1[:,:,:,30:285]) # (32, 512, 1, 255)
    # deconv1 = lib.ops.deconv1d("deconv1",deconv2,3,1,256,128,bias=True,batchnorm=True) # (32, 256, 1, 257)

    # output = rdeconv1[:,:,0,:].transpose(0,2,1)
    output = rdeconv0[:,:,0,:].transpose(0,2,1)
    return (gru[:,:,0,-1],output)


print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

sequences   = T.imatrix('sequences')
h0          = T.fmatrix('h0')
reset       = T.iscalar('reset')

input_sequences = sequences[:,]
target_sequences = sequences[:,768:]

new_h0, predicted_sequences = network(input_sequences,h0,reset)
cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(predicted_sequences.reshape((-1,Q_LEVELS))),
    target_sequences.flatten()
).mean()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.print_params_info(cost, params)
#updates = lib.optimizers.Adam(cost, params, 1e-3,gradClip=True,value=GRAD_CLIP)
grads = T.grad(cost, wrt=params)
updates = lasagne.updates.adam(grads, params, learning_rate=0.01)

print "Gradients Computed"

train_fn = theano.function(
    [sequences, h0, reset],
    [cost, new_h0,predicted_sequences],
    updates=updates,
    on_unused_input='warn'
)

input_seq = T.imatrix()
test_h0 = T.fmatrix()
test_reset = T.iscalar()

test_new_h0,test_predict = network(input_seq,test_h0,test_reset)
test_fn = theano.function(
    [input_seq, test_h0, test_reset],
    [test_new_h0,T.nnet.softmax(test_predict.reshape((-1,Q_LEVELS)))]
)

def generate_and_save_samples(tag,seed_h0):

    def write_audio_file(name, data):

        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95

        import scipy.io.wavfile
        scipy.io.wavfile.write(name+'.wav',BITRATE,data)

    # Generate 5 sample files, each 5 seconds long
    N_SEQS = 32
    LENGTH = 8*BITRATE
    LENGTH += LENGTH%31

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :SEQ_LEN] = Q_ZERO

    #if seed_h0:
    #    h0 = seed_h0
    #else:
    #    h0 = numpy.zeros((N_SEQS, DIM), dtype='float32')
    h0 = seed_h0
    frame_level_outputs = None

    for t in xrange(31, LENGTH-31,31):
        h0,probs = test_fn(samples[:,t-31:t],h0,0)
        probs = probs.reshape((N_SEQS,31,Q_LEVELS))
        samples[:,t:t+31] = np.argmax(probs,axis=2)
        print t

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i])


#grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

print "Training!"
total_iters = 0

for epoch in xrange(NB_EPOCH):
    h0 = np.zeros((BATCH_SIZE, DIM)).astype(theano.config.floatX)
    costs = []
    times = []
    data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))
#    seqs = data_feeder[20][0]
#    reset = data_feeder[20][1]

    for seqs, reset in data_feeder:
    # while True:
        start_time = time.time()
        cost, h0, _ = train_fn(seqs, h0, reset)
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters
        costs.append(cost)
        print "\tCost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
#        if total_iters%500==0:
#            generate_and_save_samples('iterno_%d'%total_iters)
