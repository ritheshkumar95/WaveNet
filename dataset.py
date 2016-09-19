import numpy
import scipy.io.wavfile
import scikits.audiolab

import random
import time
import numpy as np


random_seed = 123

def feed_epoch(data_path, n_files, BATCH_SIZE, SEQ_LEN, OVERLAP, Q_LEVELS, Q_ZERO,RF=1024):
    global random_seed
    """
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).
    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.
    Loads sequentially-named FLAC files in a directory
    (p0.flac, p1.flac, p2.flac, ..., p[n_files-1].flac)
    Assumes all flac files have the same length.
    data_path: directory containing the flac files
    n_files: how many FLAC files are in the directory
    (see two_tier.py for a description of the constants)
    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """

    def round_to(x, y):
        """round x up to the nearest y"""
        return int(numpy.ceil(x / float(y))) * y

    def mewlaw_quantize(data):
	final_data = []
	for i in xrange(data.shape[0]):
	    final_data.append(float_to_uint8(ulaw(wav_to_float(data[i]))))
        return np.asarray(final_data,dtype=np.uint8)

    def ulaw(x, u=255):
        x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
        return x

    def invulaw(y,u=255):
        y = np.sign(y)*(1./u)*(np.power(1+u,np.abs(y))-1)
        return y

    def float_to_uint8(x):
        x += 1.
        x /= 2.
        uint8_max_value = np.iinfo('uint8').max
        x *= uint8_max_value
        x = x.astype('uint8')
        return x


    def wav_to_float(x):
        try:
            max_value = np.iinfo(x.dtype).max
            min_value = np.iinfo(x.dtype).min
        except:
            max_value = np.finfo(x.dtype).max
            min_value = np.finfo(x.dtype).min
        x = x.astype('float64', casting='safe')
        x -= min_value
        x /= ((max_value - min_value) / 2.)
        x -= 1.
        return x

    def batch_quantize(data):
        """
        floats in (-1, 1) to ints in [0, Q_LEVELS-1]
        scales normalized across axis 1
        """
        eps = numpy.float64(1e-5)
        companded = np.sign(data)*(np.log(1+255*np.abs(data))/np.log(256))
        data = companded

        data -= data.min(axis=1)[:, None]

        data *= ((Q_LEVELS - eps) / data.max(axis=1)[:, None])
        data += eps/2
        # print "WARNING using zero-dc-offset normalization"
        # data -= data.mean(axis=1)[:, None]
        # data *= (((Q_LEVELS/2.) - eps) / numpy.abs(data).max(axis=1)[:, None])
        # data += Q_LEVELS/2

        data = data.astype('uint8')

        return data

    paths = [data_path+'/p{}.flac'.format(i) for i in xrange(n_files)]
    #rand_idx = np.random.randint(0,141867,n_files)
    #paths = [data_path+'/p{}.flac'.format(i) for i in rand_idx]

    random.seed(random_seed)
    random.shuffle(paths)
    random_seed += 1

    batches = []
    for i in xrange(len(paths) / BATCH_SIZE):
        batches.append(paths[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    random.shuffle(batches)

    for batch_paths in batches:
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])
        batch_seq_len = round_to(batch_seq_len, SEQ_LEN)

        batch = numpy.zeros(
            (BATCH_SIZE, batch_seq_len),
            dtype='float64'
        )

        for i, path in enumerate(batch_paths):
            data, fs, enc = scikits.audiolab.flacread(path)
            batch[i, :len(data)] = data

        if Q_LEVELS != None:
            batch = batch_quantize(batch)

            batch = numpy.concatenate([
                numpy.full((BATCH_SIZE, OVERLAP), Q_ZERO, dtype=np.uint8),
                batch
            ], axis=1)
        else:
            batch = numpy.concatenate([
                numpy.full((BATCH_SIZE, OVERLAP), 0, dtype='float32'),
                batch
            ], axis=1)
            batch = batch.astype('float32')

            batch -= batch.mean()
            batch /= batch.std()

        for i in xrange((batch.shape[1] - OVERLAP) // SEQ_LEN):
            reset = numpy.int32(i==0)
	    start = i*SEQ_LEN-RF
	    if start<0:
		start = 0
	    end = (i+1)*SEQ_LEN
            subbatch = batch[:, start : end]
            yield (subbatch, reset)
