import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import os

from rnn_cells import DecoderCell, get_cell

in_message = '\033[38;5;123m' # make all messages blue
out_message = '\033[0m' # unmake all messages blue

def model_fn(inputs, hparams, mode, labels=None, invert=None):
    assert mode in ['train', 'encode', 'decode', 'eval', 'eval_sample']
    print(in_message)
    # embedding
    if mode != 'decode':
        embedded, lengths, emb_weights = embedding(inputs, hparams)
    else:
        emb_weights = embedding(None, hparams)
    
    # encoding (for autoencoders)
    context = state = latent = loss_latent = None
    if hparams['encoder_type'] is not None \
    and mode != 'decode':
        context, state = encoder(embedded, lengths, hparams, mode)
        # latent variable
        latent, loss_latent = latent_layer(tf.concat(state, 1), hparams, mode, invert)
    
    if mode == 'encode':
        print(out_message)
        return latent
    
    # decoding
    if mode == 'decode':
        latent = inputs
        inputs = None
    probs = decoder(latent, inputs, emb_weights, hparams, mode)
    
    if mode == 'decode':
        print(out_message)
        return tf.argmax(probs, 2)
    
    # calculating loss
    with tf.variable_scope('loss'):
        loss = cross_entropy = None
        label_probs = tf.reduce_sum(probs * tf.one_hot(labels, probs.shape[2]), axis=2)
        nll = -tf.log(label_probs)
        cross_entropy = nll * lengths_to_mask(lengths + 1, tf.shape(nll)[1])
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        ppl = nll * lengths_to_mask(lengths + 1, tf.shape(nll)[1])
        ppl = tf.reduce_sum(ppl, 1)
        perplexity = tf.exp(ppl / tf.cast(lengths + 1, tf.float32))
        perplexity = tf.reduce_mean(perplexity)
        cross_entropy = tf.reduce_mean(cross_entropy)
    # losses dict
    losses = loss_latent or {}
    losses['nll'] = cross_entropy
    losses['ppl'] = perplexity
    
    print(out_message)
    return probs, losses

def latent_layer(inputs, hparams, mode, invert):
    assert mode in ['train', 'encode', 'eval', 'eval_sample']
    assert hparams['latent_type'] in ['bottleneck', 'vae', 'binary', 'gumbel', None]
    with tf.variable_scope('latent'):
        loss = {}
        if hparams['latent_type'] == 'bottleneck':
            latent = tf.layers.dense(inputs, hparams['latent_size'], use_bias=False)
            if invert is not None:
                inv_mask = tf.reshape(tf.one_hot(invert, hparams['latent_size']), [1, hparams['latent_size']])
                latent = (1 - inv_mask) * latent + inv_mask * (-latent)
        elif hparams['latent_type'] == 'vae':
            mu = tf.layers.dense(inputs, hparams['latent_size'], name='mu')
            logsigma = tf.layers.dense(inputs, hparams['latent_size'], name='logsigma')
            sigma = tf.exp(logsigma)
            # calculate kullback-leibler divergence
            loss['kl'] = (tf.reduce_sum(mu**2, 1) + tf.reduce_sum(sigma**2, 1)) / 2 - \
                    tf.reduce_sum(logsigma, 1) - hparams['latent_size'] / 2
            loss['kl'] = tf.reduce_mean(loss['kl'])
            if invert is not None:
                inv_mask = tf.reshape(tf.one_hot(invert, hparams['latent_size']), [1, hparams['latent_size']])
                mu = (1 - inv_mask) * mu + inv_mask * (-mu)
            if mode in ['train', 'eval_sample']:
                latent = mu + sigma * tf.random_normal(tf.shape(sigma))
            else:
                latent = mu
        elif hparams['latent_type'] in ['binary', 'gumbel']:
            # probability vector
            logits = tf.layers.dense(inputs, hparams['latent_size'])
            if invert is not None:
                inv_mask = tf.reshape(tf.one_hot(invert, hparams['latent_size']), [1, hparams['latent_size']])
                logits = (1 - inv_mask) * logits + inv_mask * (-logits)
            prob = tf.nn.sigmoid(logits)
            # random sampling on train, or just rounding on inference
            if mode in ['train', 'eval_sample']:
                if hparams['latent_type'] == 'binary':
                    epsilon = tf.random_uniform(tf.shape(prob), 0, 1)
                    latent = step_function(prob - epsilon)
                    latent = latent + prob - tf.stop_gradient(prob)
                else:
                    e = tf.reshape(logits, [-1, hparams['latent_size'], 1])
                    g1 = -tf.log(-tf.log(tf.random_uniform(tf.shape(e), 0, 1) + 1e-12) + 2e-12)
                    g2 = -tf.log(-tf.log(tf.random_uniform(tf.shape(e), 0, 1) + 1e-12) + 2e-12)
                    # Gumbel-softmax
                    latent = tf.nn.softmax(tf.concat([e + g1, -e + g2], 2), 2)[:, :, 0]
            else:
                latent = step_function(prob - 0.5)
                # latent = prob
                # latent = logits
            if hparams.get('use_kl', False):
                q = 1 - prob
                kl = tf.reduce_sum(prob * tf.log(prob + 1e-12) + q * tf.log(q + 1e-12) - tf.log(0.5), 1)
                loss['kl'] = tf.reduce_mean(kl)
        elif hparams['latent_type'] is None:
            latent = inputs
    return latent, loss

def embedding(inputs, hparams, emb_weights=None, name='embeddings', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if emb_weights is None:
            # Load embeddings from glove embeddings file
            vectors = []
            for i, l in enumerate(open(os.path.join(hparams['data_path'], 'embeddings.txt'))):
                if i > hparams['vocab_size']:
                    break
                vec = [float(w) for w in l.strip().split()[1:]]
                vectors.append(vec)
            vectors = [[0 for _ in vec]] * 3 + vectors # <pad>, <EOS>, <unk>
            vectors = np.array(vectors).astype('float32')
            # create variable
            emb_weights = tf.get_variable('embeddings', initializer=vectors, trainable=False)
        if inputs is not None:
            embedded = tf.nn.embedding_lookup(emb_weights, inputs)
            mask = step_function(inputs, tf.int32)
            lengths = tf.reduce_sum(mask, 1)
    if inputs is None:
        return emb_weights
    return embedded, lengths, emb_weights

def decoder(latent, inputs, emb_weights, hparams, mode, name='decoder', reuse=False):
    assert mode in ['train', 'decode', 'eval', 'eval_sample']
    with tf.variable_scope(name, reuse=reuse):
        # Make cell
        decoder_cell = DecoderCell(latent, hparams, mode)
        if latent is None:
            batch_size = tf.shape(inputs)[0]
        else:
            batch_size = tf.shape(latent)[0]
        initial_state = decoder_cell.zero_state(batch_size, tf.float32)
        def softmax_wo_unk(x):
            # this is a hack. makes the decoder not produce unks all the time
            return tf.nn.softmax(x) * (1 - tf.one_hot([2], emb_weights.shape[0]))
        projection_layer = layers_core.Dense(emb_weights.shape[0], use_bias=False,
                activation=softmax_wo_unk if mode=='decode' else tf.nn.softmax,
                name='projection')
        
        # Decode at training
        if mode in ['train', 'eval', 'eval_sample']:
            # Helper
            helping_inputs = tf.concat([tf.fill([batch_size, 1], 0), inputs], 1)
            embedded, lengths, _ = embedding(helping_inputs, hparams, emb_weights)
            helper = tf.contrib.seq2seq.TrainingHelper(
                    embedded, lengths + 1, time_major=False)
            # Decoder
            seq_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, initial_state,
                    output_layer=projection_layer)
            # Dynamic decoding
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(seq_decoder)
            probabilities = outputs.rnn_output
        
        # Decode at inference
        if mode == 'decode':
            # Helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    emb_weights,
                    tf.fill([batch_size], 0), 1)
            # Decoder
            seq_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, initial_state,
                    output_layer=projection_layer)
            # Dynamic decoding
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    seq_decoder,
                    maximum_iterations=hparams['max_output_length'])
            probabilities = outputs.rnn_output
    
    return probabilities

def encoder(inputs, lengths, hparams, mode, name='encoder', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        n = hparams['encoder_layers']
        layer_configs = {
            'rnn': [rnn_layer] * n,
            'rnn_bidi': [bidi_rnn_layer] * n,
            'rnn_bidi_onlyfirst': [bidi_rnn_layer] + [rnn_layer] * (n - 1),
            'transformer': [positional_embeddings] + [transformer_layer] * n,
            'rnn_transformer': [bidi_rnn_layer] + [transformer_layer] * (n - 1),
            'cnn': [positional_embeddings] + [cnn_layer] * n,
            'rnn_cnn': [bidi_rnn_layer] + [cnn_layer] * (n - 1),
            'cnn_transformer': [positional_embeddings] + [(cnn_layer, transformer_layer)] * n,
            'rnn_cnn_transformer': [bidi_rnn_layer] + [(cnn_layer, transformer_layer)] * (n - 1),
        }
        layers = layer_configs[hparams['encoder_type']]
        state = []
        context = inputs
        # Sequence of layers
        for layer in layers:
            if type(layer) == tuple:
                # Multiple layers in parallel
                ctxs = []
                for sublayer in layer:
                    ctx, st = sublayer(context, lengths, hparams, mode)
                    if st is not None:
                        state.append(st)
                    ctxs.append(ctx)
                context = tf.math.add_n(ctxs)
                context = tf.contrib.layers.layer_norm(context)
            else:
                # Just one layer
                context, st = layer(context, lengths, hparams, mode)
                if st is not None:
                    state.append(st)
    return context, state

# bidirectional recurrent layer
def bidi_rnn_layer(inputs, lengths, hparams, mode, name='bidi_rnn_layer'):
    with tf.variable_scope(name):
        (context_fw, context_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            get_cell(hparams['encoder_cell'], hparams, mode),
            get_cell(hparams['encoder_cell'], hparams, mode),
            inputs,
            dtype=tf.float32,
            sequence_length=lengths
        )
        context = tf.concat([context_fw, context_bw], 2)
        state = tf.concat([state_fw, state_bw], 1)
    return context, state

# recurrent layer
def rnn_layer(inputs, lengths, hparams, mode, name='rnn_layer'):
    with tf.variable_scope(name):
        context, state = tf.nn.dynamic_rnn(
            get_cell(hparams['encoder_cell'], hparams, mode),
            inputs,
            dtype=tf.float32,
            sequence_length=lengths
        )
    return context, state

# add positional embeddings to input
def positional_embeddings(inputs, lengths, hparams, mode, name='positional_embeddings'):
    with tf.variable_scope(name):
        pe = tf.get_variable('pe', [1, 1, hparams['max_output_length']])
        context = tf.contrib.layers.layer_norm(inputs + pe[:, :, :inputs.shape[2]])
    return context, None

# transformer layer from "Attention is all you need" but with gelu activation (like in BERT)
def transformer_layer(inputs, lengths, hparams, mode, name='transformer_layer'):
    model_dim = hparams['hidden_size']
    inner_dim = hparams['transformer_inner_dim']
    num_heads = hparams['transformer_heads']
    assert model_dim % num_heads == 0
    with tf.variable_scope(name):
        print(name)
        # add state input
        inputs = tf.concat([tf.get_variable('state_emb', [1, 1, int(inputs.shape[2])]), inputs], 1)
        # make mask
        bs = tf.shape(inputs)[0] # batch size
        length = tf.shape(inputs)[1]
        mask = lengths_to_mask(lengths + 1, length) # +1 for state
        dk = model_dim // num_heads
        # masked multi-head attention
        inputs = dropout(inputs, hparams, mode)
        Q = tf.layers.dense(inputs, model_dim, use_bias=False)
        K = tf.layers.dense(inputs, model_dim, use_bias=False)
        V = tf.layers.dense(inputs, model_dim, use_bias=False)
        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        att = Q @ tf.transpose(K, [0, 2, 1]) / np.sqrt(dk)
        att = tf.nn.softmax(att, axis=1)
        mask = tf.concat([tf.reshape(mask, [bs, length, 1])] * num_heads, 0)
        att = att * mask
        att = att / (tf.reduce_sum(att, 1, keepdims=True) + 1e-8)
        att = att @ V
        att = tf.concat(tf.split(att, num_heads, axis=0), axis=2)
        att = tf.reshape(att, [bs, length, model_dim])
        # add & norm
        if inputs.shape[2] == model_dim:
            att = tf.contrib.layers.layer_norm(att + inputs)
        else:
            print('projecting residual in layer "%s"' % name)
            att = tf.contrib.layers.layer_norm(att \
                    + tf.layers.dense(inputs, model_dim, use_bias=False,
                    name='projection_fix'))
        # feed forward
        att_res = att
        att = dropout(att, hparams, mode)
        h = tf.layers.dense(att, inner_dim)
        h = h * 0.5 * (1 + tf.erf(h / np.sqrt(2))) # GELU activation
        h = dropout(h, hparams, mode)
        h = tf.layers.dense(h, model_dim)
        # add & norm
        h = tf.contrib.layers.layer_norm(h + att_res)
        # split
        context = h[:, 1:, :]
        state = h[:, 0, :]
    return context, state

# convolution layer with bottleneck and residual connection
def cnn_layer(inputs, lengths, hparams, mode, name='conv_layer'):
    with tf.variable_scope(name):
        print(name)
        bs = tf.shape(inputs)[0] # batch size
        length = tf.shape(inputs)[1]
        mask = tf.reshape(tf.range(length), [1, length]) - np.array(0.5)
        mask = tf.cast(mask, tf.float32)
        mask = (tf.cast(tf.sign(tf.cast(tf.reshape(lengths,
                [bs, 1]), tf.float32) - mask), tf.float32) + 1) / 2 # [bs, length]
        # conv operation
        x = dropout(inputs, hparams, mode)
        x = tf.layers.dense(x, hparams['hidden_size'] // 2, activation=tf.nn.relu)
        x = dropout(x, hparams, mode)
        x = x * tf.reshape(mask, [bs, length, 1])
        x = tf.layers.conv1d(x, hparams['hidden_size'] // 2, 3, activation=tf.nn.relu, padding='same')
        x = dropout(x, hparams, mode)
        conv = tf.layers.dense(x, hparams['hidden_size'], activation=tf.nn.relu)
        # residual connection
        if inputs.shape[2] == model_dim:
            conv = tf.contrib.layers.layer_norm(conv + inputs)
        else:
            print('projecting residual in layer "%s"' % name)
            conv = tf.contrib.layers.layer_norm(conv \
                    + tf.layers.dense(inputs, model_dim, use_bias=False,
                    name='projection_fix'))
        state = tf.contrib.layers.layer_norm(tf.reduce_sum(conv, 1))
    return conv, state


def dropout(x, hparams, mode):
    if mode == 'train':
        return tf.nn.dropout(x, keep_prob=1 - hparams['dropout_rate'])
    return x

# heaviside step function
def step_function(inputs, dtype=None):
    dtype = dtype or inputs.dtype
    x = (tf.sign(inputs) + 1) / 2
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def lengths_to_mask(lengths, length=None):
    if length is None:
        length = tf.reduce_max(lengths)
    mask = tf.reshape(tf.range(length), [1, length]) - np.array(0.5)
    mask = tf.cast(mask, tf.float32)
    lengths = tf.cast(tf.reshape(lengths, [-1, 1]), tf.float32)
    mask = step_function(lengths - mask)
    return mask
