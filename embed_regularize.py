"""Implementation of the dropout Embedding based on keras Embedding"""
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from absl import logging


class DropoutEmbedding(layers.Embedding):
  """Turns positive integers (indexes) into dense vectors of fixed size,
  with a dropout mask.

  e.g. `[[4], [20], [57]] -> [[0.25, 0.1], [0.6, -0.2], [0.9, 0.86]]` ->
  `[[0.25, 0.1], [0.6, -0.2], [0.0, 0.0]]`

  This layer can only be used as the first layer in a model.

  Arguments:
    training: Whether in the training mode.
    dropout: Float between 0 and 1.
      Fraction of the words to drop for
      the embedding transformation of the one-hot embedding.

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.
  """

  def __init__(self, training, dropout=0., **kwargs):
    kwargs['dtype'] = 'float32'
    super(DropoutEmbedding, self).__init__(**kwargs)
    self.training = training
    self.dropout = dropout
    if 0. < self.dropout < 1.:
      self.uses_learning_phase = True

  def call(self, inputs):
    if K.dtype(inputs) != 'int32':
      inputs = K.cast(inputs, 'int32')
    if self.dropout < 0. or self.dropout > 1.:
      logging.warning('WARNING: value of dropout not in [0, 1), '
                      'automatically set to 0.')
      self.dropout = 0.
    if 0. < self.dropout < 1. and self.training:
      retain_p = 1. - self.dropout
      self.B = K.random_binomial((self.input_dim,), p=retain_p) * \
               (1. / retain_p)
      self.B = K.expand_dims(self.B)
      self.W = self.embeddings * self.B
    else:
      self.W = self.embeddings
    out = K.gather(self.W, inputs)
    return out
