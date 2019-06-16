import tensorflow as tf
import numpy as np
from multiprocessing_generator import ParallelGenerator as PG
import os
from tqdm import tqdm
import json

from featurizer import batch_generator
from model import model_fn
from my_utils import *

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

# Model
inputs = tf.placeholder(tf.int32, [None, None])
labels = tf.placeholder(tf.int32, [None, None])
sample_count = tf.placeholder(tf.float32, [])
probs, losses = model_fn(inputs, hparams, 'train', labels)
loss = losses['nll']
if hparams.get('latent_type', None) == 'vae':
    # KL loss annealing
    kl_koef = 0.01 + tf.reduce_min([sample_count / hparams['kl_annealing_period'], 0.99])
    loss += losses['kl'] * kl_koef
    losses['kl_koef'] = kl_koef

# Optimizer and gradient clipping
optimizer = tf.train.AdamOptimizer(learning_rate=hparams['learning_rate'])
gvs = optimizer.compute_gradients(loss)
with tf.variable_scope('clipping'):
    gvs = [(None if grad is None else tf.clip_by_norm(grad, 0.2),
            var) for grad, var in gvs]

# Training operator
tf.set_random_seed(0)
train_op = optimizer.apply_gradients(gvs, tf.train.get_global_step())
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create folder if necessary
if not os.path.exists(hparams['model_path']):
    os.makedirs(hparams['model_path'])
train_metrics = open(os.path.join(hparams['model_path'], 'train_metrics.txt'), 'w')
dev_metrics = open(os.path.join(hparams['model_path'], 'dev_metrics.txt'), 'w')

# Train model on dataset
with tf.Session() as sess:
    sess.run(init_op)
    # Train multiple epochs
    num_batches = 0
    sample_count_int = 0
    for epoch in range(hparams['num_epochs']):
        print('Epoch', epoch)
        # Train
        metrics = {}
        # Get batches from generator
        with PG(batch_generator(hparams, 'train'), 10) as g:
            if epoch != 0:
                g = GeneratorLen(g, num_batches) # use progressbar
            for step, (batch_x, batch_y) in enumerate(tqdm(g)):
                # Train one step
                _, ls = sess.run((train_op, losses), {
                    inputs: batch_x,
                    labels: batch_y,
                    sample_count: sample_count_int})
                sample_count_int += len(batch_x)
                # add loss to array
                for l in ls:
                    metrics[l] = metrics.get(l, []) + [ls[l]]
                # count batches
                if epoch == 0:
                    num_batches += 1
        # print average losses
        metrics = {l: float(np.mean(metrics[l])) for l in metrics}
        train_metrics.write(json.dumps(metrics) + '\n')
        print('train loss:', metrics)
        # Evaluate on dev
        metrics = {}
        # Get batches from generator
        with PG(batch_generator(hparams, 'dev'), 10) as g:
            for step, (batch_x, batch_y) in enumerate(g):
                dev_losses = {l: losses[l] for l in losses if l != 'kl_koef'}
                ls = sess.run(dev_losses, {
                    inputs: batch_x,
                    labels: batch_y})
                # add loss to array
                for l in ls:
                    metrics[l] = metrics.get(l, []) + [ls[l]]
        # print average losses
        metrics = {l: float(np.mean(metrics[l])) for l in metrics}
        dev_metrics.write(json.dumps(metrics) + '\n')
        print('dev loss:', metrics)
        if epoch == 0:
            print('Note: dev loss is calculated in training mode (dropouts, random latents sampling).')
    # Save weights
    save_path = saver.save(sess, os.path.join(hparams['model_path'], 'model.ckpt'))
    print("Model saved in path: %s" % save_path)












