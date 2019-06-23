from matplotlib import pyplot as plt
import json
import os
import sys
import numpy as np
import tensorflow as tf

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

vectors = np.array([[float(w) for w in l.strip().split()] for l in open(
    os.path.join(hparams['output_path'], 'vectors_energy.txt'))])
print('vectors.shape', vectors.shape)
vectors = (vectors - np.mean(vectors)) / np.std(vectors)

classes = np.array([int(l.strip()) for l in open(os.path.join(hparams['data_path'], 'test_theme.txt'))])
names = [
    'Society',
    'Science',
    'Health',
    'Education',
    'Computers',
    'Sports',
    'Business',
    'Entertainment',
    'Family',
    'Politics'
]
classes = classes - 1
print('classes', classes)

X_train = vectors[:9000, :]
y_train = classes[:9000]
X_dev = vectors[9000:, :]
y_dev = classes[9000:]

# Neural network
np.random.seed(0)
tf.reset_default_graph()
tf.set_random_seed(0)

x_in = tf.placeholder(tf.float32, [None, hparams['latent_size']])
labels = tf.placeholder(tf.int32, [None])
dropout_bool = tf.placeholder(tf.float32, [])
dropout = lambda x: x * (1 - dropout_bool) + dropout_bool * tf.nn.dropout(x, keep_prob=0.7)
x = x_in
x = tf.layers.dense(x, 100, activation=tf.nn.tanh)
x = dropout(x)
x = tf.layers.dense(x, 100, activation=tf.nn.tanh)
x = dropout(x)
prob = tf.layers.dense(x, 10, activation=tf.nn.softmax)
cls = tf.argmax(prob, 1)
loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(labels, 10) * tf.log(prob + 1e-12), 1))

# Optimizer with gradient clipping
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
gvs = optimizer.compute_gradients(loss)
gvs = [(grad if grad is None else tf.clip_by_norm(grad, 0.2), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(gvs, tf.train.get_global_step())

train_loss = []
dev_loss = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    # initialize weights
    sess.run(init)
    for epoch_i in range(100):
        l, _ = sess.run([loss, train_op], {x_in: X_train, labels: y_train, dropout_bool: 1})
        train_loss.append(l)
        l, cl = sess.run([loss, cls], {x_in: X_dev, labels: y_dev, dropout_bool: 0})
        dev_loss.append(l)
        acc = [1 if c == ref else 0 for (c, ref) in zip(cl, y_dev)]
        acc = sum(acc) / len(acc)
print('NLL:', dev_loss[-1])
print('Accuracy:', acc)

fout = open(os.path.join(hparams['output_path'], 'classification_loss.txt'), 'w')
for l in dev_loss:
    fout.write('%f\n' % l)

# Draw plot
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.set_xlabel('Epoch')
axis.set_ylabel('loss')
axis.grid(linestyle='dotted')
axis.tick_params(direction='in')
axis.plot(train_loss, label='train')
axis.plot(dev_loss, label='dev')
axis.legend()
plt.show()
#fig.show()
#plt.savefig('paper/clusters.png')
