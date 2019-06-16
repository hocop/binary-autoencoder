import tensorflow as tf
import numpy as np
from multiprocessing_generator import ParallelGenerator as PG
import os
from tqdm import tqdm
import json
import gc

from featurizer import batch_generator
from model import model_fn
from my_utils import *

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

metrics = {}

# Count batches
print('Counting batches')
num_batches = 0
for _ in batch_generator(hparams, 'test'):
    num_batches += 1
print(num_batches, 'batches')

for mode in ['eval', 'eval_sample']:
    # Model
    tf.reset_default_graph()
    tf.set_random_seed(0)
    inputs = tf.placeholder(tf.int32, [None, None])
    labels = tf.placeholder(tf.int32, [None, None])
    probs, losses = model_fn(inputs, hparams, mode, labels)
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # This kostyl helps to run evaluation second time with different graph
    gc.collect()
    
    # Evaluate model on test set
    with tf.Session() as sess:
        # Load model
        saver.restore(sess, os.path.join(hparams['model_path'], 'model.ckpt'))
        # Get batches from generator
        with PG(batch_generator(hparams, 'test'), 10) as g:
            g = GeneratorLen(g, num_batches) # use progressbar
            for batch_x, batch_y in tqdm(g):
                for _ in range(1 if mode == 'eval' else 10):
                    ls = sess.run(losses, {inputs: batch_x, labels: batch_y})
                    # add loss to array
                    for l in ls:
                        ll = l
                        if mode == 'eval_sample':
                            ll += '_s'
                        metrics[ll] = metrics.get(ll, []) + [ls[l]]

# print average losses
metrics = {l: float(np.mean(metrics[l])) for l in metrics}
print('test loss:', metrics)

# Create folder if necessary
if not os.path.exists(hparams['output_path']):
    os.makedirs(hparams['output_path'])
with open(os.path.join(hparams['output_path'], 'test_metrics.txt'), 'w') as f:
    f.write(json.dumps(metrics))
