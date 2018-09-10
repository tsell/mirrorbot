from __future__ import print_function

import numpy as np
import tensorflow as tf

from util import constants as const
from model.saveable_model import SaveableModel

class ActionResponseNetwork(object):
  """The model described in Figure XX.
  
  Given a VideoStream and an ActionController, learns to predict the
  effects of the ActionController's actions on the VideoStream's video.
  """

  def __init__(self, args):
    """Our model takes in two tensors:
        1. A batch of frames, with this shape:
            [imwidth, imheight, imchannels, batchsize]
        2. An integer representing an action.
      And outputs a predicted next frame and new action weights."""
    self.inputs(args)
    self.embedding(args)
    self.preprocess(args)
    self.rnnlayers(args)
    self.optimizer(args)

  def inputs(self, args):
    # Our inputs.
    with tf.variable_scope('inputs') as scope:
      self.action = tf.get_variable('action', shape=[1])
      self.prior_iterator_handle = tf.placeholder(
          tf.string, shape=[], name='prior_iterator')
      self.response_iterator_handle = tf.placeholder(
          tf.string, shape=[], name='response_iterator')

      # Iterates over the prior window.
      prior_iterator = tf.data.Iterator.from_string_handle(
          self.prior_iterator_handle,
          const.FRAME_DTYPE,
          shape=[FRAME_BATCH_SIZE] + const.FRAME_SHAPE)

      # Iterates over the response window.
      response_iterator = tf.data.Iterator.from_string_handle(
          self.response_iterator_handle,
          const.FRAME_DTYPE,
          shape=[FRAME_BATCH_SIZE] + const.FRAME_SHAPE)

  def embedding(self, args):
    with tf.variable_scope('embedding') as scope:
      # The action embedding.
      embed_matrix = tf.get_variable(
          'embed_matrix', 
          shape=[const.ACTION_VOCAB_SIZE, const.ACTION_EMBED_SIZE],
          initializer=tf.random_uniform_initializer(),
          trainable=True)
      action_embedding = tf.nn.embedding_lookup(
          embed_matrix,
          action_input,
          name='action_embedding')

  def preprocess(self, args):
    with tf.variable_scope('preprocess') as scope:
      # A mask of potentially-affected pixels.
      mask = tf.get_variable(
          'mask',
          const.FRAME_DTYPE,
          shape=[FRAME_BATCH_SIZE] + const.FRAME_SHAPE)

      # A 1-D vector of all the pixels in the prior frame(s).
      flat_prior= tf.reshape(self.prior_iterator.get_next(), [-1])

      # Concatenate all inputs into a big 1-D vector.
      self.flat_input = tf.concat(action_embedding, flat_prior)

  def rnnlayers(self, args):
    pass

  def predictions(self, args):
    # Apply the RNN on all inputs.
    # This predicts the response frame.
    flat_logits = self.rnn.apply(inputs=self.rnn1)

  def optimizer(self, args)
    # Compute loss as the difference between expected and actual response.
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=flat_logits,
          labels=flat_labels))

    # Optimize toward the minimal difference between expected and actual frames.
    opt = tf.train.AdamOptimizer(
        learning_rate=const.LEARNING_RATE, global_step=global_step)
    self.optimizer = opt.minimize(self.loss)

  
  def run(self):
    with tf.Session() as sess:
      step = 0
      while True:
        step += 1
        predicted_response, next_action = sess.run(
          self.predicted_response_and_next_action,
          feed_dict={
            action: next_action,
            prior_iterator: 
            }))
        if step % 1000 == 0:
          saver.Save(sess, 
