import lib
import theano
import numpy as np
import theano.tensor as T

def RMSprop(cost, params, learnrate, rho=0.90, epsilon=1e-6):
    gparams = []
    iter = 1
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)  
        print param['name'] + " completed"
    updates=[]
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * gparam ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        gparam = gparam / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((param, param - gparam * learnrate))
    return updates

def Adam(cost, params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,gradClip=True,value=1.):
    gparams = []
    iter = 1
    for param in params:
        gparam = T.grad(cost,param)
        if gradClip:
    	   gparam = T.clip(gparam,lib.floatX(-value), lib.floatX(value))
    	gparams.append(gparam)
    	print str(iter) + " completed"
    	iter += 1
    updates = []
    for p, g in zip(params, gparams):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * (g ** 2)
        
        gradient_scaling = T.sqrt(v_new + epsilon)
        updates.append((m, m_new))
        updates.append((v, v_new))
        updates.append((p, p - lr * m / gradient_scaling))
    return updates
