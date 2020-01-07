"""Training and testing the ordered-neuron lstm model using the PTB dataset"""
from absl import app
from absl import flags
from absl import logging

import os
import time
import tensorflow as tf # version:1.13.1
tf.enable_v2_behavior()
from tensorflow.python.keras import optimizers
import numpy as np
import rnn_model

EVAL_BATCH_SIZE = 1

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './data/train.txt',
                    'location of the training data')
flags.DEFINE_string('valid_dir', './data/valid.txt',
                    'location of the validation data')
flags.DEFINE_string('test_dir', './data/test.txt', 'location of the test data')
flags.DEFINE_enum('model', 'lstm', ['on_lstm', 'lstm', 'gru'],
                  'type of recurrent net (on_lstm, lstm, gru)')
flags.DEFINE_integer('vocab_size', 10000, 'size of vocabulary')
flags.DEFINE_integer('embed_size', 650, 'size of word embeddings')
flags.DEFINE_integer('hidden_units', 650, 'number of hidden units per layer')
flags.DEFINE_integer('chunk_size', 10, 'number of units per chunk')
flags.DEFINE_integer('layer_num', 2, 'number of layers')
flags.DEFINE_float('lr', 5.0, 'initial learning rate')
flags.DEFINE_float('clip', 0.25, 'gradient clipping')
flags.DEFINE_integer('epochs', 200, 'upper epoch limit')
flags.DEFINE_integer('batch_size', 20, 'training batch size')
flags.DEFINE_integer('buffer_size', 10000, 'buffer size')
flags.DEFINE_integer('seq_length', 35, 'sequence length')
flags.DEFINE_float('dropout', 0.65,
                   'dropout applied to layers (0 = no dropout)')
flags.DEFINE_float('rnn_dropout', 0.25,
                   'dropout for rnn layers (0 = no dropout)')
flags.DEFINE_float('input_dropout', 0.5,
                   'dropout for input embedding layers (0 = no dropout)')
flags.DEFINE_float(
    'embed_dropout', 0.1, 'dropout to remove words from embedding'
    'layers (0 = no dropout)')
flags.DEFINE_float(
    'w_dropout', 0.2, 'amount of weight dropout to apply to '
    'the RNN hidden to hidden matrix')
flags.DEFINE_float(
    'alpha', 1.0, 'alpha L2 regularization on RNN activation '
    '(alpha = 0 means no regularization)')
flags.DEFINE_float(
    'beta', 1.0, 'beta slowness regularization applied on RNN '
    'activiation (beta = 0 means no regularization)')
flags.DEFINE_boolean('tied', False,
                     'whether or not to tie the embedding weight')
flags.DEFINE_float('w_decay', 1.2e-6, 'weight decay applied to all weights')
flags.DEFINE_integer('seed', 1111, 'random seed')
flags.DEFINE_integer('when', 20000, 'when will the lr decay by 10')
flags.DEFINE_float('lr_decay', 0.25, 'the decay rate of lr')
flags.DEFINE_string('save_dir', './model', 'path to save the final model')
flags.DEFINE_integer('measurement_store_interval', 1,
                     'The number of steps between storing objective value in '
                     'measurements.')
flags.DEFINE_string('logdir', '/tmp/on_lstm',
                    'Path to directory where to store summaries.')


def load_data(data_dir, mode='train'):
  """Read the data from dir."""
  filein = open(data_dir, 'rb').read().decode(encoding='utf-8')
  text = []
  for line in filein.strip().split('\n'):
    sent = line.split() + ['<eos>']
    text += sent
  logging.info('Length of {} text: {} words'.format(mode, len(text)))
  return text


def dataset_generate(text, word2idx, batch_size):
  """Reshape the loaded data according to batch_size."""
  text_as_int = np.array([word2idx[w] for w in text])
  n_batch = len(text_as_int) // batch_size
  dataset = tf.constant(text_as_int[:batch_size * n_batch])
  dataset = tf.reshape(dataset, [batch_size, -1])

  return dataset


def dataset_preparation(train_dir, valid_dir, test_dir):
  """Creating a mapping from unique words to indices."""
  train_text = load_data(train_dir, mode='train')
  valid_text = load_data(valid_dir, mode='valid')
  test_text = load_data(test_dir, mode='test')

  # Creating a mapping from unique words to indices in training mode
  # Note that due to the PTB dataset itself, the vocabulary in validation
  # and test set has already been covered by the training set, there is no
  # need to deal with the OOV.
  word_dict = {}
  for w in train_text:
    if w not in word_dict:
      word_dict[w] = 1
    else:
      word_dict[w] += 1
  word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
  vocab = [k for k, v in word_dict]
  logging.info('Number of words: {} unique words'.format(len(vocab)))
  word2idx = {u: i for i, u in enumerate(vocab)}
  idx2word = np.array(vocab)
  train_dataset = dataset_generate(train_text, word2idx, batch_size=FLAGS.batch_size)
  valid_dataset = dataset_generate(valid_text, word2idx, batch_size=EVAL_BATCH_SIZE)
  test_dataset = dataset_generate(test_text, word2idx, batch_size=EVAL_BATCH_SIZE)

  return train_dataset, valid_dataset, test_dataset, vocab


def get_batch(source, i, batch_size):
  """Get the input and the target data at each time step"""
  seq_len = min(FLAGS.seq_length, tf.shape(source)[1] - 1 - i)
  data = tf.slice(source, [0, i], [batch_size, seq_len])
  target = tf.slice(source, [0, i+1], [batch_size, seq_len])

  return data, target


def train_step(inp, target, model, optimizer):
  """Train the model: use tf.GradientTape to track the gradients."""
  with tf.GradientTape() as tape:
    predictions, hidden, dropped_hidden = model(inp)
    # tf.print(dropped_hidden, output_stream=sys.stderr)
    raw_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        target, predictions))
    loss = raw_loss

    # Activiation Regularization
    if FLAGS.alpha > 0:
      # dropped_hidden: [batch_size, seq_length, hidden_units]
      ar = tf.math.reduce_mean(
          FLAGS.alpha * tf.math.reduce_mean(
              tf.pow(dropped_hidden, 2), 2))
      loss = raw_loss + ar

    # Temporal Activation Regularization (slowness)
    if FLAGS.beta > 0:
      # hidden: [batch_size, seq_length, hidden_units]
      tar = tf.math.reduce_mean(
          FLAGS.beta * tf.math.reduce_mean(
              tf.pow(hidden[:, 1:, :] - hidden[:, :-1, :], 2), 2))
      loss = raw_loss + tar

    gvs = zip(
        tape.gradient(loss, model.trainable_variables),
        model.trainable_variables)
    capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
    optimizer.apply_gradients(capped_gvs)

    return loss, raw_loss


def main(argv):
  train_dataset, valid_dataset, test_dataset, vocab = \
    dataset_preparation(FLAGS.train_dir, FLAGS.valid_dir, FLAGS.test_dir)
  # Length of the vocabulary in words
  vocab_size = len(vocab)

  # Set random seed
  tf.random.set_random_seed(FLAGS.seed)

  # Build the model
  model_args = dict(
      model=FLAGS.model,
      layer_num=FLAGS.layer_num,
      chunk_size=FLAGS.chunk_size,
      vocab_size=vocab_size,
      embedding_dim=FLAGS.embed_size,
      rnn_units=FLAGS.hidden_units,
      embed_dropout=FLAGS.embed_dropout,
      input_dropout=FLAGS.input_dropout,
      dropout=FLAGS.dropout,
      rnn_dropout=FLAGS.rnn_dropout,
      w_dropout=FLAGS.w_dropout,
      w_decay=FLAGS.w_decay,
      tied=FLAGS.tied
  )
  model = rnn_model.RNNModel(training=True,
                             batch_size=FLAGS.batch_size,
                             **model_args)
  model.build(input_shape=(FLAGS.batch_size, None))
  # Build the model for evaluation
  eval_model = rnn_model.RNNModel(training=False,
                                  batch_size=EVAL_BATCH_SIZE,
                                  **model_args)
  eval_model.build(input_shape=(EVAL_BATCH_SIZE, None))
  # Set the learning rate scheduler
  lr_schedule = tf.train.exponential_decay(
    learning_rate=FLAGS.lr,
    global_step=0,
    decay_steps=FLAGS.when,
    decay_rate=FLAGS.lr_decay,
    staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_schedule)
  store_ppl = float('inf')
  best_epoch = None
  for epoch in range(1, FLAGS.epochs + 1):
    logging.info('=============== Epoch {} ==============='.format(epoch))
    start = time.time()
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(FLAGS.logdir, 'ckpt_{epoch}')

    # initializing the hidden state at the start of every epoch
    # initially hidden is None
    model.reset_states()
    total_train_loss = 0.0
    total_train_raw_loss = 0.0
    for batch_n, i in enumerate(range(0, tf.shape(train_dataset)[1]-1,
                                      FLAGS.seq_length)):
      inp, target = get_batch(train_dataset, i, batch_size=FLAGS.batch_size)
      train_loss, train_raw_loss = train_step(inp, target, model, optimizer)
      total_train_loss += train_loss
      total_train_raw_loss += train_raw_loss
      if batch_n > 10:
        break

    train_loss = total_train_loss / (batch_n + 1)
    train_raw_loss = total_train_raw_loss / (batch_n + 1)
    train_ppl = tf.exp(train_raw_loss)

    if epoch % FLAGS.measurement_store_interval == 0:
      logging.info('Begin Validation...')
      eval_model.set_weights(model.get_weights())
      eval_model.reset_states()
      total_valid_loss = 0.0
      for batch_n, i in enumerate(range(0, tf.shape(valid_dataset)[1]-1,
                                        FLAGS.seq_length)):
        inp, target = get_batch(valid_dataset, i, EVAL_BATCH_SIZE)
        predictions, hidden, dropped_hidden = eval_model(inp)
        total_valid_loss += tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions))
     
      valid_loss = total_valid_loss / (batch_n + 1)
      valid_ppl = tf.exp(valid_loss)
      if valid_ppl < store_ppl:
        store_ppl = valid_ppl
        best_epoch = epoch
        if epoch > 0.5*FLAGS.epochs:
          model.save_weights(checkpoint_prefix.format(epoch=epoch))


      logging.info('Begin Test...')
      eval_model.reset_states()
      total_test_loss = 0.0
      for batch_n, i in enumerate(range(0, tf.shape(test_dataset)[1]-1,
                                        FLAGS.seq_length)):
        inp, target = get_batch(test_dataset, i, EVAL_BATCH_SIZE)
        predictions, hidden, dropped_hidden = eval_model(inp)
        total_test_loss += tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions))

      test_loss = total_test_loss / (batch_n + 1)
      test_ppl = tf.exp(test_loss)


      logging.info('\nEpoch {} | '
                   'Train Loss {:.4f} | '
                   'Train PPL {:.4f} | '
                   'Valid PPL {:.4f} | '
                   'Min Valid PPL {:.4f}| '
                   'Best Epoch {}'.format(epoch, train_loss, train_ppl,
                                          valid_ppl, store_ppl, best_epoch))
      logging.info('Time taken for a epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
  app.run(main)
