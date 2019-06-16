import tensorflow as tf
import numpy as np
from multiprocessing_generator import ParallelGenerator as PG
import os
from tqdm import tqdm
import sys

from featurizer import batch_generator
from model import model_fn
from my_utils import *

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

# Model
inputs = tf.placeholder(tf.int32, [None, None])
latent = model_fn(inputs, hparams, 'encode')

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create folder if necessary
if not os.path.exists(hparams['output_path']):
    os.makedirs(hparams['output_path'])

# Create output file
opath = os.path.join(hparams['output_path'], 'vectors.txt')
ofile = open(opath, 'w')

# Predict latent representations
with tf.Session() as sess:
    # Load model
    saver.restore(sess, os.path.join(hparams['model_path'], 'model.ckpt'))
    # Get batches from generator
    with PG(batch_generator(hparams, 'test', keep_order=True), 10) as g:
        for batch_x, _ in tqdm(g):
            vecs = sess.run(latent, {inputs: batch_x})
            # round numbers
            if hparams['latent_type'] == 'binary':
                vecs = (vecs * 1.1).astype('int32') # *1.1 for numerical stability
            for vec in vecs:
                ofile.write(' '.join([str(val) for val in vec]) + '\n')

print('Saved as', opath)
