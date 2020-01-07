"""Implementation of the RNN model."""
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.wrappers import TimeDistributed
import on_lstm_layer
import embed_regularize
import tied_weight
from absl import app


class RNNModel(Model):
  """Implementation of the RNN model."""

  def __init__(self, model, layer_num, batch_size, chunk_size, training, vocab_size,
               embedding_dim, rnn_units, embed_dropout, input_dropout, dropout,
               rnn_dropout, w_dropout, w_decay, tied):
    """Initializes the parameters of the model.

    Args:
      model: Specify model of RNN layer (on_lstm, lstm or gru).
      layer_num: Number of RNN layers.
      chunk_size: Number of units per chunk in the RNN layers.
      training: Whether or not in the training mode (whether or
        not to apply the dropout).
      vocab_size: Size of vocabulary.
      embedding_dim: Size of word embedding.
      rnn_units: Number of hidden units per RNN layer.
      embed_dropout: Dropout to remove words from embedding layer.
      input_dropout: Dropout for input embedding layers.
      dropout: Dropout between RNN layers.
      rnn_dropout: Recurrent dropout.
      w_dropout: Amount of weight dropout to apply to the RNN hidden
        to hidden matrix.
      w_decay: Weight decay applied to all weight.
      tied: Whether or not to tie the weight between the embedding layer
        and the decoder layer.
    """

    super(RNNModel, self).__init__()
    self.model = model
    self.layer_num = layer_num
    self.batch_size = batch_size
    self.chunk_size = chunk_size
    self.training = training
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.rnn_units = rnn_units
    self.dropout = dropout
    self.rnn_dropout = rnn_dropout
    self.w_dropout = w_dropout
    self.input_dropout = input_dropout
    self.regularizer = regularizers.l2(w_decay)
    self.dropout_embedding = \
      embed_regularize.DropoutEmbedding(training=self.training,
                                        input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        embeddings_regularizer=self.regularizer,
                                        dropout=embed_dropout)
    self.tied = tied
    self.rnn_layer = [None] * layer_num
    if self.tied:
      self.rnn_layer_sizes = [rnn_units] * (layer_num - 1) + [embedding_dim]
      self.tied_to = self.dropout_embedding
    else:
      self.rnn_layer_sizes = [rnn_units] * layer_num
      self.tied_to = None
    rnn_model_args = dict(
        return_sequences=True,
        return_state=True,
        stateful=True,
        kernel_regularizer=self.regularizer,
        recurrent_regularizer=self.regularizer,
        dropout=w_dropout,
        recurrent_dropout=rnn_dropout,
        batch_input_shape=(self.batch_size, None, None))
    if self.model == 'on_lstm':
      for i in range(layer_num):
        self.rnn_layer[i] = \
          on_lstm_layer.OrderedNeuronLSTM(units=self.rnn_layer_sizes[i],
                                          chunk_size=chunk_size,
                                          **rnn_model_args)

    if self.model == 'lstm':
      for i in range(layer_num):
        self.rnn_layer[i] = \
          tf.keras.layers.LSTM(units=self.rnn_layer_sizes[i],
                               **rnn_model_args)

    if self.model == 'gru':
      for i in range(layer_num):
        self.rnn_layer[i] = \
          tf.keras.layers.GRU(units=self.rnn_layer_sizes[i],
                              **rnn_model_args)
    self.output_layer = TimeDistributed(
        tied_weight.TiedDense(units=vocab_size,
                              tied_to=self.tied_to,
                              kernel_regularizer=self.regularizer))

  def call(self, inputs):
    # [batch_size, seq_length, embedding_dim]
    embed = self.dropout_embedding(inputs)
    if self.training:
      x = K.dropout(embed, level=self.input_dropout)
      for i in range(self.layer_num):
        x, state_h, state_c = self.rnn_layer[i](x, training=self.training)
      dropped_hidden = K.dropout(x, level=self.dropout)
    else:
      x = embed
      for i in range(self.layer_num):
        x, state_h, state_c = self.rnn_layer[i](x, training=self.training)
      dropped_hidden = x
    hidden = x
    x = self.output_layer(dropped_hidden)
    output = K.softmax(x)

    return output, hidden, dropped_hidden

  def eval(self):
    self.training = False
    self.dropout_embedding.training = self.training

  def train(self):
    self.training = True
    self.dropout_embedding.training = self.training


def main(argv):
  del argv  # Unused
  model_args = dict(
      model="on_lstm",
      training=True,
      batch_size=20,
      layer_num=2,
      chunk_size=10,
      vocab_size=10000,
      embedding_dim=650,
      rnn_units=650,
      embed_dropout=0.1,
      input_dropout=0.5,
      dropout=0.65,
      rnn_dropout=0.25,
      w_dropout=0.2,
      w_decay=0.25,
      tied=False
  )
  model = RNNModel(**model_args)


if __name__ == '__main__':
  app.run(main)