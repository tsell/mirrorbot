import logging as log

import cv2
import numpy as np
import tensorflow as tf

from util import constants as const
from video.video_stream import VideoStream

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
    self._frame_shape = [args['output_height'], args['output_width'], 3]
    self.inputs(args)
    self.embedding(args)
    self.preprocess(args)
    self.outputs(args)

  def inputs(self, args):
    # Our inputs.
    with tf.variable_scope('inputs'):
      # The frame before we act.
      self.prior_frame_ph = tf.placeholder(
          name='prior_frame',
          dtype=const.FRAME_DTYPE,
          shape=self._frame_shape)

      # The action taken.
      self.prior_action_ph = tf.placeholder(
          name='prior_action',
          shape=(),
          dtype=tf.int32)

      # The frame after we act.
      self.response_frame_ph = tf.placeholder(
          name='response_frame',
          dtype=const.FRAME_DTYPE,
          shape=self._frame_shape)

  def embedding(self, args):
    with tf.variable_scope('embedding'):
      # The action embedding.
      embed_matrix = tf.get_variable(
          'embed_matrix', 
          shape=[const.ACTION_VOCAB_SIZE, np.prod(self._frame_shape)],
          initializer=tf.random_uniform_initializer(),
          trainable=True)
      self.action_embedding = tf.nn.embedding_lookup(
          embed_matrix,
          self.prior_action_ph,
          name='action_embedding')

  def preprocess(self, args):
    with tf.variable_scope('preprocess'):
      self.frame = tf.image.flip_up_down(self.prior_frame_ph)

  def outputs(self, args):
    with tf.variable_scope('outputs'):
      self.predicted_response = tf.image.convert_image_dtype(
          self.frame, dtype=tf.uint8)
      self.next_action = self.prior_action_ph
  
  def run(self, args):
    cap = VideoStream(args)
    cap.start()
    with tf.Session() as sess:
      step = 0
      next_action = 1
      while not cap.stopped:
        step += 1
        prior_frame_idx, prior_frame = cap.get_last_index_and_frame()
        response_frame_idx, response_frame = cap.get_last_index_and_frame()
        predicted_response, next_action = sess.run(
          [self.predicted_response, self.next_action],
          feed_dict={
            self.prior_frame_ph: prior_frame,
            self.prior_action_ph: next_action,
            self.response_frame_ph: response_frame
            })
        log.info(
            '%09d Next action: %s' %
            (step, const.ACTION_DICT[int(next_action)]))

        cap.record_prediction_frame(predicted_response)
