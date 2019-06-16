import tensorflow as tf
import numpy as np
from multiprocessing_generator import ParallelGenerator as PG
import os
from tqdm import tqdm

from featurizer import Encoder
from model import model_fn
from my_utils import *

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

# Model
inputs = tf.placeholder(tf.float32, [None, hparams['latent_size']])
decoded = model_fn(inputs, hparams, 'decode')
text_encoder = Encoder(hparams)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create output file
opath = os.path.join(hparams['output_path'], 'decoded.txt')
ofile = open(opath, 'w')

# Decode sequences
with tf.Session() as sess:
    # Load model
    saver.restore(sess, os.path.join(hparams['model_path'], 'model.ckpt'))
    # Get vectors from file
    for line in tqdm(GeneratorFile(os.path.join(hparams['output_path'], 'vectors.txt'))):
        # str to numbers
        vec = np.array([float(val) for val in line.strip().split()]).astype('float32')
        # Decode with batch size = 1
        ints = sess.run(decoded, {inputs: [vec]})[0]
        # Transform numbers to words
        sentence = text_encoder.decode(ints)
        ofile.write(sentence + '\n')

print('Saved as', opath)
