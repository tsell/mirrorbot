from __future__ import print_function

import numpy as np
import tensorflow as tf

class SaveableModel(object):
  """A TF model with convenience save/restore methods.
  """

  def __init__(self, args):
    self.graph = tf.Graph()
   
    with self.graph.as_default():
      # Set paths for saving graphdef and metagraphdef info.
      if args['graphdef_file']:
        self.graphdef_path = args['graphdef_file']
        self.metagraph_path = self.graphdef_path + '.meta'
      else:
        self.graphdef_path = tempfile.mkstemp(suffix='.graphdef')
        self.metagraph_path = self.graphdef_path + '.meta'
        
      if (args['graphdef_file'] and
          os.path.exists(self.graphdef_path) and
          os.path.exists(self.metagraph_path)):
        # If we were pointed at existing graphdef, load it.
        self.saver, self.sess = self.load_graph()
      else:
        # If we were not pointed at existing graphdef, construct new graph.
        self.saver, self.sess = self.new_graph(args)

  def __del__(self):
    if self.save_enabled:
      self.save_graph()

  def save_graph(self):
    tf.train.export_meta_graph(self.metagraph_path)
    self.saver.save(self.sess, self.graphdef_path)

  def load_graph(self):
    # Load frozen GraphDef protobuf from disk.
    saver = tf.train.import_meta_graph(self.metagraph_path, clear_devices=True)
    saver.restore(sess, self.graphdef_path)
    return saver, tf.Session(graph=self.graph)

  def new_graph(self, args):
    raise NotImplementedError("Graph not yet implemented.")

  def train(self, video_stream, action_controller):
    raise NotImplementedError("Training not yet implemented.")

  def predict(self, video_stream, action_controller):
    raise NotImplementedError("Prediction not yet implemented.")

  def train_and_predict(self, video_stream, action_controller):
    raise NotImplementedError("Training not yet implemented.")

