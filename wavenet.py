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

# Hyperparams
NB_EPOCH=200
BATCH_SIZE = 16
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 16
BITRATE = 16000

Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
N_BLOCKS=1
RF=N_BLOCKS*1024-N_BLOCKS+2
FRAME_SIZE = RF # How many samples per frame
SEQ_LEN=FRAME_SIZE+RF
n_filters=256
#data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))

def network(input_sequences):
    batch_size = input_sequences.shape[0]
    length = input_sequences.shape[1]
    dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*N_BLOCKS]).tolist()[0]
    # start = lib.ops.Embedding(
    #    'Embedding',
    #    Q_LEVELS,
    #    Q_LEVELS,
    #    input_sequences,
    # ).transpose(0,2,1)[:,:,None,:] #(32, 256, 1, 8960)
    start = T.extra_ops.to_one_hot(input_sequences.flatten(),nb_class=256).reshape((batch_size,length,256)).transpose(0,2,1)[:,:,None,:]
    conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,256,bias=False,batchnorm=False,pad=(0,1))[:,:,:,:length]
    prev_conv = conv1
    #prev_skip = []
    prev_skip = T.zeros_like(conv1)
    i=0
    for value in dilations:
        i+=1
        x,y = lib.ops.WaveNetConv1d("Block-%d"%i,prev_conv,2,n_filters,n_filters,bias=False,batchnorm=False,dilation=value)
        prev_skip += y
        #prev_skip += [y]
        prev_conv = x

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

test_sequences = T.imatrix()
output = network(test_sequences)
test_fn = theano.function(
    [test_sequences],
    output
)
print "Compiled!"

N_SEQS = 8
LENGTH = 8*BITRATE
samples = numpy.zeros((N_SEQS, LENGTH), dtype=np.uint8)
def generate(seed=0):
    global samples
    def write_audio_file(name, data):
        def ulaw2lin(x, u=255.):
            max_value = np.iinfo('uint8').max
            min_value = np.iinfo('uint8').min
            x = x.astype('float64', casting='safe')
            x -= min_value
            x /= ((max_value - min_value) / 2.)
            x -= 1.
            x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
            x = float_to_uint8(x)
            return x
        def float_to_uint8(x):
            x += 1.
            x /= 2.
            uint8_max_value = np.iinfo('uint8').max
            x *= uint8_max_value
            x = x.astype('uint8')
            return x


        # data = data.astype('float32')
        #
        # data -= data.min()
        # data /= data.max()
        # data -= 0.5
        # data *= 0.95
        #
        # def invulaw(y,u=255):
        #     y = np.sign(y)*(1./u)*(np.power(1+u,np.abs(y))-1)
        #     return y

        import scipy.io.wavfile
        #scipy.io.wavfile.write(name+'.wav',BITRATE,invulaw(data))
        scipy.io.wavfile.write(name+'.wav',BITRATE,ulaw2lin(data))


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

    data_feeder = list(dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES))
    #data_feeder = list(dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO))
    #data_feeder = list(vctk_dataset.feed_epoch(225, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES))
    print "File loaded"
    data = data_feeder[seed][0][:]
    samples[:, :RF] = data[:N_SEQS,:RF]

    for t in xrange(RF, LENGTH):
        probs = test_fn(samples[:,t-RF:t])
        samples[:,t] = sample(probs,0.001)
        #samples[:,t] = probs.flatten()
        print t, samples[:,t]

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i][RF:])




print "Training!"
for epoch in xrange(NB_EPOCH):
    costs = []
    times = []
    data_feeder = dataset.blizzard_feed_epoch(BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES)
    #data_feeder = vctk_dataset.feed_epoch(225, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, RF, N_FILES)
    print "Epoch : ",epoch
    total_iters = 0
    # for seqs, reset in data_feeder:
    #     start_time = time.time()
    #     cost,pred = train_fn(seqs,0.001)
    #     total_time = time.time() - start_time
    #     times.append(total_time)
    #     total_iters += 1
    #     print "Batch ",total_iters, " (Epoch %d)"%epoch
    #     costs.append(cost)
    #     print "\tCost: ",np.mean(costs)
    #     print "\tTime: ",np.mean(times)
    for seqs,reset in tqdm(data_feeder):
        start_time = time.time()
        cost,pred = train_fn(seqs,0.001)
        total_time = time.time() - start_time
        costs.append(cost)
        times.append(total_time)
    print "\tCost : ", np.mean(costs)
    print "\tTime : ", np.mean(times)
    if epoch%50==0:
        generate()
