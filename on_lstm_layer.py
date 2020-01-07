"""Implementation of the Ordered Neuron LSTM Layer based in keras LSTM Layer."""
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers.recurrent import LSTMCell
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.ops import array_ops


def cumsoftmax(x, axis=-1):
  return tf.math.cumsum(activations.softmax(x, axis=axis), axis=axis)


class OrderedNeuronLSTMCell(LSTMCell):
  """Cell class for the Ordered-Neuron LSTM layer.

  Reference: Ordered Neurons: Integrating Tree Structures into
    Recurrent Neural Networks (Shen et al., 2018)

  Arguments:
    chunk_size: Number of units per chunk.

  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self, chunk_size, **kwargs):
    super(OrderedNeuronLSTMCell, self).__init__(**kwargs)
    self.chunk_size = chunk_size
    self.n_chunk = int(self.units / self.chunk_size)


  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    self.kernel_tilde = self.add_weight(
        shape=(input_dim, self.n_chunk * 2),  # set dimension to be n_chunk
        name='kernel_tilde',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel_tilde = self.add_weight(
        shape=(self.units, self.n_chunk * 2),  # set dimension to be n_chunk
        name='recurrent_kernel_tilde',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2 + self.n_chunk * 2,), *args,
                                    **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4 + self.n_chunk * 2,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o, x_i_tilde, x_f_tilde = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c_hat = self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    # [1, n_chunk]
    i_tilde = 1 - cumsoftmax(x_i_tilde + K.dot(
        h_tm1_i, self.recurrent_kernel_tilde[:, :self.n_chunk]))
    # [1, n_chunk]
    f_tilde = cumsoftmax(x_f_tilde + K.dot(
        h_tm1_f, self.recurrent_kernel_tilde[:, self.n_chunk:]))
    # repeat each dimension chunk_size times, [1, n_chunk] to [1, units]
    i_tilde = tf.reshape(tf.tile(i_tilde, tf.constant([1, self.chunk_size])),
                         [tf.shape(i_tilde)[0], self.units])
    f_tilde = tf.reshape(tf.tile(f_tilde, tf.constant([1, self.chunk_size])),
                         [tf.shape(f_tilde)[0], self.units])
    # [1, units]
    w = f_tilde * i_tilde
    i_hat = i * w + (i_tilde - w)
    f_hat = f * w + (f_tilde - w)
    c_old = f * c_tm1 + i * c_hat
    c_new = f_hat * c_tm1 + i_hat * c_hat
    return c_old, c_new, o

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if 0 < self.dropout < 1.:
      inputs_i = inputs * dp_mask[0]
      inputs_f = inputs * dp_mask[1]
      inputs_c = inputs * dp_mask[2]
      inputs_o = inputs * dp_mask[3]
    else:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs
    k_i, k_f, k_c, k_o = array_ops.split(
        self.kernel, num_or_size_splits=4, axis=1)
    # [input_dim, n_chunk]
    k_i_tilde, k_f_tilde = array_ops.split(
        self.kernel_tilde, num_or_size_splits=2, axis=1)
    x_i = K.dot(inputs_i, k_i)
    x_f = K.dot(inputs_f, k_f)
    x_c = K.dot(inputs_c, k_c)
    x_o = K.dot(inputs_o, k_o)
    # [1, n_chunk]
    x_i_tilde = K.dot(inputs_i, k_i_tilde)
    x_f_tilde = K.dot(inputs_f, k_f_tilde)
    if self.use_bias:
      b_i = self.bias[:self.units]
      b_f = self.bias[self.units:self.units * 2]
      b_c = self.bias[self.units * 2:self.units * 3]
      b_o = self.bias[self.units * 3:self.units * 4]
      b_i_tilde = self.bias[self.units * 4:self.units * 4 + self.n_chunk]
      b_f_tilde = self.bias[self.units * 4 + self.n_chunk:]

      x_i = K.bias_add(x_i, b_i)
      x_f = K.bias_add(x_f, b_f)
      x_c = K.bias_add(x_c, b_c)
      x_o = K.bias_add(x_o, b_o)
      # [1, n_chunk]
      x_i_tilde = K.bias_add(x_i_tilde, b_i_tilde)
      x_f_tilde = K.bias_add(x_f_tilde, b_f_tilde)

    if 0 < self.recurrent_dropout < 1.:
      h_tm1_i = h_tm1 * rec_dp_mask[0]
      h_tm1_f = h_tm1 * rec_dp_mask[1]
      h_tm1_c = h_tm1 * rec_dp_mask[2]
      h_tm1_o = h_tm1 * rec_dp_mask[3]
    else:
      h_tm1_i = h_tm1
      h_tm1_f = h_tm1
      h_tm1_c = h_tm1
      h_tm1_o = h_tm1
    x = (x_i, x_f, x_c, x_o, x_i_tilde, x_f_tilde)
    h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
    c_old, c_new, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

    h = o * self.activation(c_new)
    h_old = o * self.activation(c_old)
    return h, [h, c_new]


class OrderedNeuronLSTM(RNN):
  """Long Short-Term Memory layer - Hochreiter 1997.

   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    chunk_size: Number of units per chunk.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               chunk_size,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):

    cell = OrderedNeuronLSTMCell(
        units=units,
        chunk_size=chunk_size,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(OrderedNeuronLSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(OrderedNeuronLSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(OrderedNeuronLSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


def _caching_device(rnn_cell):
  """Returns the caching device for the RNN variable.

  This is useful for distributed training, when variable is not located as same
  device as the training worker. By enabling the device cache, this allows
  worker to read the variable once and cache locally, rather than read it every
  time step from remote when it is needed.

  Note that this is assuming the variable that cell needs for each time step is
  having the same value in the forward path, and only gets updated in the
  backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
  cell body relies on any variable that gets updated every time step, then
  caching device will cause it to read the stall value.

  Args:
    rnn_cell: the rnn cell instance.
  """
  if context.executing_eagerly():
    # caching_device is not supported in eager mode.
    return None
  if not getattr(rnn_cell, '_enable_caching_device', False):
    return None
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
    logging.warn('Variable read device caching has been disabled because the '
                 'RNN is in tf.while_loop loop context, which will cause '
                 'reading stalled value in forward path. This could slow down '
                 'the training due to duplicated variable reads. Please '
                 'consider updating your code to remove tf.while_loop if '
                 'possible.')
    return None
  if rnn_cell._dtype_policy.should_cast_variables:
    logging.warn('Variable read device caching has been disabled since it '
                 'doesn\'t work with the mixed precision API. This is '
                 'likely to cause a slowdown for RNN training due to '
                 'duplicated read of variable for each timestep, which '
                 'will be significant in a multi remote worker setting. '
                 'Please consider disabling mixed precision API if '
                 'the performance has been affected.')
    return None
  # Cache the value on the device that access the variable.
  return lambda op: op.device
                                                                                
