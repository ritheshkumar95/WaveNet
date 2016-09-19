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
arr = np.arange(1,33).astype('float32').reshape((1,1,1,32))
W = np.asarray([1,1]).astype('float32').reshape((1,1,1,2))
import theano.tensor as T


arr2 = T.nnet.conv2d(arr,W,border_mode=(0,1),filter_dilation=(1,1)).eval()
arr3 = T.nnet.conv2d(arr2,W,border_mode=(1,2),filter_dilation=(1,2)).eval()
arr4 = T.nnet.conv2d(arr3,W,subsample=(1,2),filter_dilation=(1,1)).eval()

dilation = 4
arr = np.random.randn(32,64,1,3072).astype('float32')
W = np.random.randn(128,64,1,2).astype('float32')
true_conv = T.nnet.conv2d(arr,W,border_mode=(0,dilation),filter_dilation=(1,dilation))[:,:,:,:arr.shape[-1]].eval()
first_part = T.nnet.conv2d(arr[:,:,:,:dilation],np.expand_dims(W[:,:,:,0],axis=-1))
new_conv = T.nnet.conv2d(arr,W,filter_dilation=(1,dilation))
conc = T.concatenate((first_part,new_conv),axis=-1).eval()
(conc==true_conv).all()

import numpy as np
import theano
import theano.tensor as T
dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*1]).tolist()[0]
arr = np.arange(1,1026).astype('float32').reshape((1,1,1,1025))
W = np.asarray([1,1]).astype('float32').reshape((1,1,1,2))
arr = T.nnet.conv2d(arr,W)
for value in dilations:
#    arr = T.nnet.conv2d(arr,W,filter_dilation=(1,value))
    arr = T.nnet.conv2d(arr,W,subsample=(1,2))
