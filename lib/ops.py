import lib
import numpy as np
import numpy
import theano
import theano.tensor as T
theano.config.floatX='float32'
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import lasagne
import math

srng = RandomStreams(seed=234)

def BatchNorm(layer_name,input, insize, mode=0,run_mode=0, momentum=0.9, layer='default'):
    '''
    # params :
    input_shape :
        when mode is 0, we assume 2D input. (mini_batch_size, # features)
        when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
    mode :
        0 : feature-wise mode (normal BN)
        1 : window-wise mode (CNN mode BN)
    momentum : momentum for exponential average
    '''
    input_shape = input.shape
    # random setting of gamma and beta, setting initial mean and std
    rng = np.random.RandomState(int(time.time()))

    gamma_val = np.asarray(rng.uniform(low=-1.0/math.sqrt(insize), high=1.0/math.sqrt(insize), size=(insize)),dtype=theano.config.floatX)
    if layer=='recurrent':
        gamma = lib.param(layer_name+'.gamma', np.full(shape=(insize),fill_value=0.1,dtype=theano.config.floatX), borrow=True)
    else:
        gamma = lib.param(layer_name+'.gamma', gamma_val, borrow=True)
    beta = lib.param(layer_name+'.beta',np.zeros((insize), dtype=theano.config.floatX), borrow=True)
    mean = lib.param(layer_name+'.mean',np.zeros((insize),dtype=theano.config.floatX),  train=False, borrow=True)
    var = lib.param(layer_name+'.var',np.ones((insize), dtype=theano.config.floatX),train = False, borrow=True)

    epsilon = 1e-06

    if mode==0 :
        if run_mode==0 :
            now_mean = T.mean(input, axis=0)
            now_var = T.var(input, axis=0)
            now_normalize = (input - now_mean) / T.sqrt(now_var+epsilon) # should be broadcastable..
            output = gamma * now_normalize + beta
            # mean, var update
            # run_mean = theano.clone(mean,share_inputs=False)
            # run_var = theano.clone(var, share_inputs=False)
            # run_mean.default_update = momentum * mean + (1.0-momentum) * now_mean
            # run_var.default_update = momentum * var + (1.0-momentum) * (input_shape[0]/(input_shape[0]-1)*now_var)
            mean = momentum*mean + (1.0-momentum) * now_mean
            var = momentum*var + (1.0-momentum)*(input_shape[0]/(input_shape[0]-1))*now_var
        else :
            output = gamma * (input - mean) / T.sqrt(var+epsilon) + beta

    else :
        # in CNN mode, gamma and beta exists for every single channel separately.
        # for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
        # then, each channel has own scalar gamma/beta parameters.
        axes = (0,2,3)
        if run_mode==0 :
            now_mean = T.mean(input, axis=axes)
            now_var = T.var(input, axis=axes)
            # mean, var update
            # run_mean = theano.clone(mean,share_inputs=False)
            # run_var = theano.clone(var, share_inputs=False)
            # run_mean.default_update = momentum * mean + (1.0-momentum) * now_mean
            # run_var.default_update = momentum * var + (1.0-momentum) * (input_shape[0]/(input_shape[0]-1)*now_var)
            mean = momentum*mean + (1.0-momentum) * now_mean
            var = momentum*var + (1.0-momentum)*(input_shape[0]/(input_shape[0]-1))*now_var
        else :
            now_mean = mean
            now_var = var
        # change shape to fit input shape

        param_axes = iter(range(input.ndim - len(axes)))
        pattern = ['x' if input_axis in axes
               else next(param_axes)
               for input_axis in range(input.ndim)]
        now_mean = now_mean.dimshuffle(pattern)
        now_var = now_var.dimshuffle(pattern)
        now_gamma = gamma.dimshuffle(pattern)
        now_beta = beta.dimshuffle(pattern)
        output = now_gamma * (input - now_mean) / T.sqrt(now_var+epsilon) + now_beta

    return output.astype('float32')

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def glorot_uniform(shape,init='glorot'):
    def uniform(shape, scale=0.05, name=None):
        return np.random.uniform(low=-scale, high=scale, size=shape)
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    if init=='he':
        s = np.sqrt(6./fan_in)
        return uniform(shape,s)
    else:
        return uniform(shape, s)

def init_weights(fan_in,fan_out,init='he'):

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    if init == 'lecun' or (init == None and fan_in != fan_out):
        weight_values = uniform(numpy.sqrt(1. / fan_in), (fan_in, fan_out))

    elif init == 'he':
        weight_values = uniform(numpy.sqrt(2. / fan_in), (fan_in, fan_out))

    elif init == 'orthogonal' or (init == None and fan_in == fan_out):
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], numpy.prod(shape[1:]))
            # TODO: why normal and not uniform?
            a = numpy.random.normal(0.0, 1.0, flat_shape)
            u, _, v = numpy.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype(theano.config.floatX)
        weight_values = sample((fan_in, fan_out))
    return weight_values

def Dense(name, input_dim, output_dim, inputs, bias=True, init=None, weightnorm=True,hidden_dim=None):

    weight_values = init_weights(input_dim,output_dim,init)

    weight = lib.param(
        name + '.W',
        weight_values
    )

    batch_size = None
    if inputs.ndim==3:
        batch_size = inputs.shape[0]
        inputs = inputs.reshape((-1,input_dim))

    if weightnorm:
        norm_values = numpy.linalg.norm(weight_values, axis=0)
        norms = lib.param(
            name + '.g',
            norm_values
        )

        normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
        result = T.dot(inputs, normed_weight)

    else:
        result = T.dot(inputs, weight)

    if bias:
        b = lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        )
        result += b

    result.name = name+".output"
    if batch_size!=None:
        return result.reshape((batch_size,hidden_dim,output_dim))
    else:
        return result

def Embedding(name, n_symbols, output_dim, indices):
    vectors = lib.param(
        name,
        numpy.random.randn(
            n_symbols,
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = tuple(list(indices.shape) + [output_dim])

    return vectors[indices.flatten()].reshape(output_shape)

def softmax_and_sample(logits):
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
    samples = T.cast(
        srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
        theano.config.floatX
    ).reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)

def GRUStep(name, input_dim, hidden_dim, x_t, h_tm1):
    processed_input = lib.ops.Dense(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        x_t
    )

    gates = T.nnet.sigmoid(
        lib.ops.Dense(
            name+'.Recurrent_Gates',
            hidden_dim,
            2 * hidden_dim,
            h_tm1,
            bias=False
        ) + processed_input[:, :2*hidden_dim]
    )

    update = gates[:, :hidden_dim]
    reset  = gates[:, hidden_dim:]

    scaled_hidden = reset * h_tm1

    candidate = T.tanh(
        lib.ops.Dense(
            name+'.Recurrent_Candidate',
            hidden_dim,
            hidden_dim,
            scaled_hidden,
            bias=False,
            init='orthogonal'
        ) + processed_input[:, 2*hidden_dim:]
    )

    one = lib.floatX(1.0)
    return (update * candidate) + ((one - update) * h_tm1)

def GRU(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    inputs = inputs.transpose(1,0,2)

    def step(x_t, h_tm1):
        return GRUStep(
            name+'.Step',
            input_dim,
            hidden_dim,
            x_t,
            h_tm1
        )

    outputs, _ = theano.scan(
        step,
        sequences=[inputs],
        outputs_info=[h0],
    )

    out = outputs.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def recurrent_fn(x_t, h_tm1,name,input_dim,hidden_dim,W1,b1,W2,b2):
    A1 = T.nnet.sigmoid(BatchNorm(name+".Inp2Hid",T.dot(x_t,W1[:input_dim]),2*hidden_dim,layer='recurrent') +
                        BatchNorm(name+".Hid2Hid",T.dot(h_tm1,W1[input_dim:]),2*hidden_dim,layer='recurrent') + b1)

    #A1 = T.nnet.sigmoid(T.dot(T.concatenate((x_t,h_tm1),axis=1),W1) + b1)

    z = A1[:,:hidden_dim]

    r = A1[:,hidden_dim:]

    scaled_hidden = r*h_tm1

    h = T.tanh(BatchNorm(name+".Candidate",T.dot(T.concatenate((scaled_hidden,x_t),axis=1),W2),hidden_dim,layer='recurrent')+b2)

    # h = T.tanh(T.dot(T.concatenate((scaled_hidden,x_t),axis=1),W2)+b2)

    one = lib.floatX(1.0)
    return ((z * h) + ((one - z) * h_tm1)).astype('float32')

def myGRU(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    inputs = inputs.transpose(1,0,2)

    weight_values = lasagne.init.GlorotUniform().sample((input_dim+hidden_dim,2*hidden_dim))
    W1 = lib.param(
        name+'.Gates.W',
        weight_values
    )

    b1 = lib.param(
        name+'.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    weight_values = lasagne.init.GlorotUniform().sample((input_dim+hidden_dim,hidden_dim))
    W2 = lib.param(
        name+'.Candidate.W',
        weight_values
    )

    b2 = lib.param(
        name+'.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    def step(x_t, h_tm1):
        return recurrent_fn(
            x_t,
            h_tm1,
            name,
            input_dim,
            hidden_dim,
            W1,b1,W2,b2
        )

    outputs, _ = theano.scan(
        step,
        sequences=[inputs],
        outputs_info=[h0],
    )

    out = outputs.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out


def recurrent_fn_hred(x_t, h_tm1,hidden_dim,W1,b1,W2,b2):
    global DIM
    #A1 = T.nnet.sigmoid(lib.ops.BatchNorm(T.dot(T.concatenate((x_t,h_tm1),axis=1),W1),name="FrameLevel.GRU"+str(name)+".Input.",length=2*512) + b1)
    A1 = T.nnet.sigmoid(T.dot(T.concatenate((x_t,h_tm1),axis=1),W1) + b1)

    z = A1[:,:hidden_dim]

    r = A1[:,hidden_dim:]

    scaled_hidden = r*h_tm1

    #h = T.tanh(lib.ops.BatchNorm(T.dot(T.concatenate((scaled_hidden,x_t),axis=1),W2),name="FrameLevel.GRU"+str(name)+".Output.",length=512)+b2)
    h = T.tanh(T.dot(T.concatenate((scaled_hidden,x_t),axis=1),W2) + b2)

    one = lib.floatX(1.0)
    return ((z * h) + ((one - z) * h_tm1)).astype('float32')

def HRED_GRU(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    global DIM
    inputs = inputs.transpose(1,0,2)

    weight_values = init_weights(input_dim+hidden_dim,2*hidden_dim)

    s_W1 = lib.param(
        'Session.Gates.W',
        weight_values
    )

    s_b1 = lib.param(
        'Session.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    weight_values = init_weights(input_dim+hidden_dim,hidden_dim)
    s_W2 = lib.param(
        'Session.Candidate.W',
        weight_values
    )

    s_b2 = lib.param(
        'Session.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    weight_values = init_weights(input_dim+hidden_dim,2*hidden_dim)

    W1 = lib.param(
        name+'.Gates.W',
        weight_values
    )

    b1 = lib.param(
        name+'.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    weight_values = init_weights(input_dim+hidden_dim,hidden_dim)
    W2 = lib.param(
        name+'.Candidate.W',
        weight_values
    )

    b2 = lib.param(
        name+'.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    outputs, _ = theano.scan(
        recurrent_fn_hred,
        sequences=[inputs],
        outputs_info=[T.alloc(0,inputs.shape[1],hidden_dim).astype(dtype=theano.config.floatX)],
        non_sequences=[hidden_dim,W1,b1,W2,b2]
    )

    #out = recurrent_fn(outputs[-1],h0,hidden_dim,s_W1,s_b1,s_W2,s_b2,"0")
    out = recurrent_fn(outputs[-1],h0,hidden_dim,s_W1,s_b1,s_W2,s_b2)

    #DIM=hidden_dim
    #out = outputs.dimshuffle(1,0,2)
    #out.name = name+'.output'
    return out


def conv1d(name,input,kernel,stride,n_filters,depth,bias=False,batchnorm=False,pad='valid',filter_dilation=(1,1),run_mode=0):
    W = lib.param(
        name+'.W',
        lasagne.init.HeNormal().sample((n_filters,depth,1,kernel)).astype('float32')
        )

    out = T.nnet.conv2d(input,W,subsample=(1,stride),border_mode=pad,filter_dilation=filter_dilation)

    if bias:
        b = lib.param(
            name + '.b',
            np.zeros(n_filters).astype('float32')
            )

        out += b[None,:,None,None]

    if batchnorm:
        out = BatchNorm(name,out,n_filters,mode=1,run_mode=run_mode)

    return out

def ResNetConv1d(name,input,kernel,stride,n_filters,depth,bias=False,batchnorm=False):
    if stride==1 and n_filters==depth:
        project = input
    else:
        project = lib.ops.conv1d(name+".Projection.conv",input,1,stride,n_filters,depth,bias=bias,batchnorm=batchnorm)
    pad = (kernel-1)/2
    conv1 = T.nnet.relu(lib.ops.conv1d(name+".conv1",input,kernel,stride,n_filters,depth,bias=bias,batchnorm=batchnorm,pad=(0,pad)))
    conv2 = lib.ops.conv1d(name+".conv2",conv1,kernel,1,n_filters,n_filters,bias=bias,batchnorm=batchnorm,pad=(0,pad))

    out = T.nnet.relu(conv2+project)
    return out

def WaveNetConv1d(name,input,kernel,n_filters,depth,bias=False,batchnorm=False,dilation=1):
    conv1 = lib.ops.conv1d(name+".filter",input,kernel,1,n_filters,depth,bias,batchnorm,pad=(0,dilation),filter_dilation=(1,dilation))[:,:,:,:input.shape[-1]]
    conv2 = lib.ops.conv1d(name+".gate",input,kernel,1,n_filters,depth,bias,batchnorm,pad=(0,dilation),filter_dilation=(1,dilation))[:,:,:,:input.shape[-1]]
    z = T.tanh(conv1)*T.nnet.sigmoid(conv2)
    out = lib.ops.conv1d(name+".projection",z,1,1,depth,n_filters,bias=bias,batchnorm=batchnorm)
    return out+input,out

def WaveNetGenConv1d(name,input,kernel,n_filters,depth,bias=False,batchnorm=False,run_mode=1):
    conv1 = lib.ops.conv1d(name+".filter",input,kernel,2,n_filters,depth,bias,batchnorm)
    conv2 = lib.ops.conv1d(name+".gate",input,kernel,2,n_filters,depth,bias,batchnorm)
    z = T.tanh(conv1)*T.nnet.sigmoid(conv2)
    out = lib.ops.conv1d(name+".projection",z,1,1,depth,n_filters,bias=bias,batchnorm=batchnorm)
    return out+input[:,:,:,1::2],out


def ResNetDeconv1d(name,input,kernel,stride,n_filters,depth,bias=False,batchnorm=False,act=True):
    if stride==1 and n_filters==depth:
        project = input
    else:
        project = lib.ops.deconv1d(name+".Projection.conv",input,1,stride,n_filters,depth,bias=bias,batchnorm=batchnorm,output=stride*input.shape[-1])
    pad = (kernel-1)/2

    conv2 = T.nnet.relu(lib.ops.deconv1d(name+".conv2",input,kernel,1,n_filters,depth,bias=bias,batchnorm=batchnorm,output=input.shape[-1],pad=(0,pad)))
    conv1 = T.nnet.relu(lib.ops.deconv1d(name+".conv1",conv2,kernel,stride,n_filters,n_filters,bias=bias,batchnorm=batchnorm,output=stride*conv2.shape[-1],pad=(0,pad)))

    if act:
        out = T.nnet.relu(conv1+project)
    else:
        out = conv1+project
    return out

def deconv1d(name,input,kernel,stride,n_filters,depth,bias=False,batchnorm=False,output=None,pad='valid'):
    if output:
        o = output
        if kernel==3:
            if stride==1:
                o += 2
            if stride==2:
                o += 1
    else:
        o = output = stride*(input.shape[-1]-1) + kernel

    W = lib.param(
        name+'.W',
        lasagne.init.HeNormal().sample((depth,n_filters,1,kernel)).astype('float32')
        )

    ### Changing this to fit kernel 3
    out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(output_grad=input,filters=W,input_shape=(None,n_filters,1,o),subsample=(1,stride))[:,:,:,:output]

    if bias:
        b = lib.param(
            name + '.b',
            np.zeros(n_filters).astype('float32')
            )

        out += b[None,:,None,None]

    if batchnorm:
        out = BatchNorm(name,out,n_filters,mode=1)

    return out

def pool(input):
    x = input[:,:,0,:]
    return x.reshape((-1,x.shape[1]*4,x.shape[2]/4))[:,:,None,:]

def subsample(input):
    x = input[:,:,0,:]
    idx = T.arange(0,x.shape[2],4)
    return x[:,:,idx][:,:,None,:]

def upsample(input):
    x = input[:,:,0,:]
    return x.reshape((-1,x.shape[1]/4,x.shape[2]*4))[:,:,None,:]
