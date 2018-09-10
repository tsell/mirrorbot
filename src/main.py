#!/usr/bin/env python

from __future__ import print_function

from model.action_response import ActionResponseNetwork
from video.video_stream import VideoStream
from action.default_action_controller import DefaultActionController

import argparse
import os

# Disable Tensorflow warnings about CPU features
# with which it could have been compiled.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__=='__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-c', '--checkpoint', type=str, default='model/frozen_graph.pb',
      help='Path to frozen GraphDef checkpoint protobuf.')
  ap.add_argument('-n', '--num_frames', type=int, default=0,
      help='# of frames to process, 0 for unlimited.')
  ap.add_argument('-q', '--nodisplay', action='store_false',
      help='Disable live display of video. Only takes effect if num_frames != 0.')
  ap.add_argument('-od', '--output_dir', type=str, default=None,
      help='Output directory path for videos.')
  ap.add_argument('-v', '--verbose', default=0, type=int, action='count',
      help='Verbosity level.')
  ap.add_argument('-fbs', '--frame_buffer_size', type=int, default=12,
      help='Max number of frames to keep in buffer.')

  args = vars(ap.parse_args())

  vid = VideoStream(args)
  actor = DefaultActionController(args)
  net = ActionResponseNetwork(args)

  if args['num_frames'] == 0:
    net.train_and_predict(vid, actor)
  else:
    net.train(vid, actor)
    net.predict(vid, actor)
