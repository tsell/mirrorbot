#!/usr/bin/env python

import logging

from model.action_response import ActionResponseNetwork

import argparse
import multiprocessing
import os

# Disable Tensorflow warnings about CPU features
# with which it could have been compiled.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TMP_PATH = '/tmp/robot_self_awareness'
os.makedirs(TMP_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=('%s/%d.log' % (TMP_PATH, os.getpid()))
    )
logging.getLogger().addHandler(logging.StreamHandler())

if __name__=='__main__':
  multiprocessing.set_start_method('spawn')
  ap = argparse.ArgumentParser()
  ap.add_argument('-c', '--checkpoint', type=str, default='model/frozen_graph.pb',
      help='Path to frozen GraphDef checkpoint protobuf.')
  ap.add_argument('-n', '--num_frames', type=int, default=0,
      help='# of frames to process, 0 for unlimited.')
  ap.add_argument('-q', '--nodisplay', action='store_true', default=False,
      help='Disable live display of video.')
  ap.add_argument('-nr', '--norecord', action='store_true', default=False,
      help='Disable recording webcame input.')
  ap.add_argument('-s', '--video_source', type=int, default=0,
      help='Video input source (number)')
  ap.add_argument('-oh', '--output_height', type=int, default=640,
      help='Video output height.')
  ap.add_argument('-ow', '--output_width', type=int, default=480,
      help='Video output width.')
  ap.add_argument('-od', '--output_dir', type=str, default=TMP_PATH,
      help='Output directory path for videos.')
  ap.add_argument('-v', '--verbose', default=0, action='count',
      help='Verbosity level.')
  ap.add_argument('-fbs', '--frame_buffer_size', type=int, default=12,
      help='Max number of frames to keep in buffer.')

  args = vars(ap.parse_args())

  net = ActionResponseNetwork(args)
  net.run(args)
