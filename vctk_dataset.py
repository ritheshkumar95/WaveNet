import numpy
import scipy.io.wavfile
import scikits.audiolab
import scipy.signal
import random
import time
import numpy as np
import glob


random_seed = 123

def feed_epoch(speaker_id,BATCH_SIZE, SEQ_LEN, STRIDE, RF=1025, N_FILES=None):
    global random_seed
    def process_wav(desired_sample_rate, filename, use_ulaw):
        channels = scipy.io.wavfile.read(filename)
        file_sample_rate, audio = channels
        audio = ensure_mono(audio)
        audio = wav_to_float(audio)
        if use_ulaw:
            audio = ulaw(audio)
        audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio)
        audio = float_to_uint8(audio)
        return audio


    def ulaw(x, u=255):
        x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
        return x


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
        x = x.astype('float32', casting='safe')
        x -= min_value
        x /= ((max_value - min_value) / 2.)
        x -= 1.
        return x


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

    def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
        if file_sample_rate != desired_sample_rate:
            mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
        return mono_audio


    def ensure_mono(raw_audio):
        """
        Just use first channel.
        """
        if raw_audio.ndim == 2:
            raw_audio = raw_audio[:, 0]
        return raw_audio

    DATA_PATH = "/tmp/kumarrit/vctk/VCTK-Corpus/wav48/p" + str(speaker_id) + "/*"
    paths = glob.glob(DATA_PATH)
    if N_FILES:
        paths = paths[:N_FILES]
    random_seed += 1
    batches = []
    for i in xrange(len(paths) / BATCH_SIZE):
        batches.append(paths[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    random.shuffle(batches)
    for batch_paths in batches:
        data = []
        for fname in batch_paths:
            data.append(process_wav(16000,fname,True))
        max_len = max([len(vec) for vec in data])
        for i in xrange(len(data)):
            data[i] = np.hstack((data[i],np.full(max_len-len(data[i]),128,dtype=np.uint8)))
        data = np.asarray(data).astype(np.uint8)
        for i in xrange(0,data.shape[1]-RF-STRIDE,STRIDE):
            start = i
            end = i+RF+STRIDE
            subbatch = data[:, start : end]
            yield (subbatch,reset)
