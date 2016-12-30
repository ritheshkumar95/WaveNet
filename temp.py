import theano
import numpy as np
import theano.tensor as T
arr = np.arange(1,33).astype('float32').reshape((1,1,1,32))
filt = np.asarray([1,2,3]).astype('float32').reshape((1,1,1,3))
filt2 = np.asarray([1]).astype('float32').reshape((1,1,1,1))

arr = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(output_grad=arr,filters=filt,input_shape=(None,1,1,arr.shape[-1]*2+1),subsample=(1,2)).eval()

filt = np.asarray([1,1,1]).astype('float32').reshape((1,1,1,3))
arr = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(output_grad=arr,filters=filt,input_shape=(None,1,1,arr.shape[-1]+1)).eval()

#arr = T.nnet.conv2d(arr,filt,subsample=(1,2),border_mode=(0,1)).eval()
#arr = T.concatenate((arr[:,:,:,0][:,:,:,None],arr),axis=-1).eval()


import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal.pool

def unpool1d(input,upsample,desired_length,pool_indices=None):
    out = T.extra_ops.repeat(input,upsample,axis=2)[:,:,:desired_length]
    if pool_indices:
        mask = T.lt(pool_indices,0)
        return mask*out
    return out

dilations = np.asarray([[1,2,4,8,16,32]*1]).tolist()[0]
length=6000
N_BLOCKS=2
arr = -1*np.arange(1,length+1).astype('float32').reshape((1,1,length,1))
W = np.asarray([0,1]).astype('float32').reshape((1,1,2,1))
indices=[]
for j in xrange(N_BLOCKS):
    for value in dilations:
    #    arr = T.nnet.conv2d(arr,W,filter_dilation=(1,value))
        arr = T.nnet.conv2d(arr,W,filter_dilation=(value,1),border_mode=(value,0))[:,:,:length]
    arr,idx = lib.ops.pool1d(arr,4,3,True)
    indices+=[idx]
    length=arr.shape[2]
for j in xrange(N_BLOCKS):
    leng = indices[-(j+1)].shape[2]
    arr = unpool1d(arr,4,leng,indices[-(j+1)])

((36*4+32)*4+32)*4+32

l=49
arr = T.as_tensor_variable(np.random.randint(0,256,(1,1,l,1)).astype('float32'))
#arr = T.as_tensor_variable(np.arange(1,l+1).astype('float32').reshape((1,1,l,1)))
(res,idx) = lib.ops.pool1d(arr,4,3,True)
#res = lib.ops.pool1d(arr,4,3,False)
res2 = T.as_tensor_variable(np.random.randint(0,256,res.shape.eval()))
out = lib.ops.unpool1d(res,4,l,idx).eval()
#out = lib.ops.unpool1d(res,4,None).eval()
print res.shape.eval()
print arr.eval().flatten()
print res.eval().flatten()
print idx.eval().flatten()
#print res2.eval().flatten()
print out.flatten()

import lib
import lib.ops
import numpy as np
import theano
import theano.tensor as T
input_sequences = T.imatrix()
Q_LEVELS=256
n_filters=64
length = input_sequences.shape[1]
start =  (input_sequences.astype('float32')/lib.floatX(Q_LEVELS-1) - lib.floatX(0.5))[:,None,None,:]
conv1 = lib.ops.conv1d("causal-conv",start,2,1,n_filters,1,bias=False,batchnorm=False,pad=(0,1))[:,:,:,:length]
f = theano.function([input_sequences],[conv1])
