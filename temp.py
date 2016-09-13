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

dilations = np.asarray([[1,2,4,8,16,32,64,128,256,512]*3]).tolist()[0]
for value in dilations:
    arr = T.nnet.conv2d(arr,w,filter_dilation=(1,value))
arr.eval()
