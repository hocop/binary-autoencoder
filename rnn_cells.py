import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.layers import base as base_layer

# returns cell by its name
def get_cell(cell_type, hparams, mode, i=0):
    # Choose cell by name
    assert cell_type in ['gru', 'dwsgru', 'indgru', 'cnn']
    if cell_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(hparams['hidden_size'])
    elif cell_type == 'dwsgru':
        cell = DepthwiseSeparableGRUCell(hparams['hidden_size'])
    elif cell_type == 'indgru':
        cell = IndRNNCell(hparams['hidden_size'])
    elif cell_type == 'cnn':
        cell = DilatedCNNCell(hparams['hidden_size'], hparams['decoder_dilation'][i], mode)
    
    # Residual connection
    cell = tf.nn.rnn_cell.ResidualWrapper(cell, my_residual_fn)
    # Dropout
    keep_prob = 1 - hparams['dropout_rate'] if mode == 'train' else 1
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell

class DecoderCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, latent, hparams, mode,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None,
            name=None,
            dtype=None):
        super(DecoderCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        self.hparams = hparams
        self.latent = latent
        self.mode = mode
        
        # init inner cell
        layers = [get_cell(hparams['decoder_cell'], hparams, mode,i) for i in range(hparams['decoder_layers'])]
        self.cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    
    @property
    def state_size(self):
        shape = self.cell.state_size
        print('Decoder state shape', shape)
        return shape
    
    @property
    def output_size(self):
        return self.hparams['hidden_size']
    
    def zero_state(self, batch_size, dtype):
        z = self.cell.zero_state(batch_size, dtype)
        if self.hparams['encoder_type'] is not None \
        and self.hparams['decoder_cell'] in ['gru', 'dwsgru', 'indgru']:
            print('Passing latent as initial decoder state')
            z = tuple([tf.layers.dense(self.latent, st.shape[1]) for st in z])
        else:
            print('Using zeros as initial decoder state')
        return z
    
    def build(self, inputs_shape):
        pass
    
    def call(self, inputs, state_in):
        # word dropout
        if self.mode == 'train' and self.hparams.get('word_dropout', False):
            bs = tf.shape(inputs)[0]
            wd = (tf.cast(tf.sign(tf.random_uniform([bs, 1]) - self.hparams['word_dropout']), tf.float32) + 1) / 2
            inputs *= wd
        # concatenate latent to inputs
        if self.hparams['concat_latent_to_words']:
            if self.hparams['encoder_type'] is None:
                raise BaseError('Do not use option "concat_latent_to_words" when there is no encoder.')
            print('Concatenating latent vector to each decoder step input')
            inputs = tf.concat([inputs, self.latent], 1)
        # call rnn-cell
        output, new_state = self.cell(
                inputs,
                state=state_in)
        return output, new_state

# my cell DWSGRU
class DepthwiseSeparableGRUCell(tf.nn.rnn_cell.RNNCell):
    # Depthwise separable RNN
    def __init__(self,
            num_units,
            activation=None,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None,
            name=None,
            dtype=None):
        super(DepthwiseSeparableGRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self.num_units = num_units
        self.activation = activation or tf.nn.tanh
    
    @property
    def state_size(self):
        return self.num_units
    
    @property
    def output_size(self):
        return self.num_units
    
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)
        input_depth = inputs_shape[1].value
        self.input_depth = input_depth
        self.kernel = tf.get_variable('kernel', [1, self.num_units, 2, 3])
        self.bias = tf.get_variable('bias', [1, self.num_units, 3])
        self.candidate_bias = tf.get_variable('candidate_bias', [1, self.num_units])
        self.built = True
    
    def call(self, inputs, state_in):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self.input_depth != self.num_units:
            inputs = tf.layers.dense(inputs, self.num_units)
        inp = tf.concat([
                tf.reshape(inputs, [-1, self.num_units, 1, 1]),
                tf.reshape(state_in, [-1, self.num_units, 1, 1])
            ], 2) # [-1, self.num_units, 2, 1]
        # RNN
        outp = tf.reduce_sum(inp * self.kernel, 2) + self.bias # [-1, self.num_units, 3]
        reset, update, candidate = tf.split(outp, 3, axis=2) # [-1, self.num_units, 1]
        reset = tf.nn.sigmoid(tf.reshape(reset, [-1, self.num_units]))
        update = tf.nn.sigmoid(tf.reshape(update, [-1, self.num_units]))
        candidate = tf.reshape(candidate, [-1, self.num_units])
        candidate = self.activation(candidate * reset + self.candidate_bias)
        
        new_state = state_in * (1 - update) + candidate * update
        
        # Dense + activation
        output = tf.layers.dense(new_state, self.num_units, activation=self.activation)
        
        return output, new_state

# dilated convolution as recurrent cell, for text generation
class DilatedCNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
            num_units,
            dilation,
            mode,
            activation=None,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None,
            name=None,
            dtype=None):
        super(DilatedCNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self.num_units = num_units
        self.dilation = dilation
        self.activation = activation or tf.nn.relu
        self.mode = mode
    
    @property
    def state_size(self):
        return (self.num_units // 2) * max(2, self.dilation)
    
    @property
    def output_size(self):
        return self.num_units
    
    def build(self, inputs_shape):
        pass
    
    def call(self, inputs, state_in):
        # bottleneck
        x = tf.layers.dense(inputs, self.num_units // 2, activation=self.activation)
        # "convolution"
        if self.dilation == 1:
            conv_in = tf.concat([state_in, x], 1)
        else:
            conv_in = tf.concat([state_in[:, :self.num_units // 2], x], 1)
        if self.mode == 'train':
            conv_in = tf.nn.dropout(conv_in, keep_prob=0.9)
        conv = tf.layers.dense(conv_in, self.num_units // 2, activation=self.activation)
        # bottleneck out
        if self.mode == 'train':
            conv = tf.nn.dropout(conv, keep_prob=0.9)
        output = tf.layers.dense(conv, self.num_units, activation=self.activation)
        # shift state
        new_state = tf.concat([state_in[:, self.num_units // 2:], x], 1)
        return output, new_state

def my_residual_fn(inp, outp):
    if inp.shape[1] != outp.shape[1]:
        print('Projecting residual in residual wrapper')
        inp = tf.layers.dense(inp, int(outp.shape[1]), use_bias=False)
    return tf.contrib.layers.layer_norm(inp + outp)

# indrnn. code stolen from https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py
class IndRNNCell(tf.nn.rnn_cell.RNNCell):
    """Independently RNN Cell. Adapted from `rnn_cell_impl.BasicRNNCell`.
    Each unit has a single recurrent weight connected to its last hidden state.
    The implementation is based on:
        https://arxiv.org/abs/1803.04831
    Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
    "Independently Recurrent Neural Network (IndRNN): Building A Longer and
    Deeper RNN"
    The default initialization values for recurrent weights, input weights and
    biases are taken from:
        https://arxiv.org/abs/1504.00941
    Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
    "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
    Args:
        num_units: int, The number of units in the RNN cell.
        recurrent_min_abs: float, minimum absolute value of each recurrent weight.
        recurrent_max_abs: (optional) float, maximum absolute value of each
            recurrent weight. For `relu` activation, `pow(2, 1/timesteps)` is
            recommended. If None, recurrent weights will not be clipped.
            Default: None.
        recurrent_kernel_initializer: (optional) The initializer to use for the
            recurrent weights. If None, every recurrent weight is initially set to 1.
            Default: None.
        input_kernel_initializer: (optional) The initializer to use for the input
            weights. If None, the input weights are initialized from a random normal
            distribution with `mean=0` and `stddev=0.001`. Default: None.
        activation: Nonlinearity to use.    Default: `relu`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.    If not `True`, and the existing scope already has
            the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                             num_units,
                             recurrent_min_abs=0,
                             recurrent_max_abs=None,
                             recurrent_kernel_initializer=None,
                             input_kernel_initializer=None,
                             activation=None,
                             reuse=None,
                             name=None):
        super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._recurrent_min_abs = recurrent_min_abs
        self._recurrent_max_abs = recurrent_max_abs
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._activation = activation or nn_ops.relu

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value
        if self._input_initializer is None:
            self._input_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                                                                                     stddev=0.001)
        self._input_kernel = self.add_variable(
                "input_kernel",
                shape=[input_depth, self._num_units],
                initializer=self._input_initializer)

        if self._recurrent_initializer is None:
            self._recurrent_initializer = init_ops.constant_initializer(1.)
        self._recurrent_kernel = self.add_variable(
                "recurrent_kernel",
                shape=[self._num_units],
                initializer=self._recurrent_initializer)

        # Clip the absolute values of the recurrent weights to the specified minimum
        if self._recurrent_min_abs:
            abs_kernel = math_ops.abs(self._recurrent_kernel)
            min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
            self._recurrent_kernel = math_ops.multiply(
                    math_ops.sign(self._recurrent_kernel),
                    min_abs_kernel
            )

        # Clip the absolute values of the recurrent weights to the specified maximum
        if self._recurrent_max_abs:
            self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                                                                            -self._recurrent_max_abs,
                                                                                                            self._recurrent_max_abs)

        self._bias = self.add_variable(
                "bias",
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Run one time step of the IndRNN.
        Calculates the output and new hidden state using the IndRNN equation
            `output = new_state = act(W * input + u (*) state + b)`
        where `*` is the matrix multiplication and `(*)` is the Hadamard product.
        Args:
            inputs: Tensor, 2-D tensor of shape `[batch, num_units]`.
            state: Tensor, 2-D tensor of shape `[batch, num_units]` containing the
                previous hidden state.
        Returns:
            A tuple containing the output and new hidden state. Both are the same
                2-D tensor of shape `[batch, num_units]`.
        """
        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.multiply(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output
