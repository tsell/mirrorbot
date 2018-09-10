"""
Video in/out utility.
"""
import datetime
from multiprocessing import Process

import cv2
import numpy as np

class VideoStream(object):
  def __init__(self, args):
    # Store start/end time and frame count, for computing actual FPS.
    self._start_time = None
    self._end_time = None
    self._frame_count = 0

    # Store whether the process should be interrupted.
    self._stopped = False
    
    # Define video input from webcam or other input source.
    self.src = args['video_source'] or 0
    self.video_in = cv2.VideoCapture(self.src)

    # Define video codec and output writer.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    resolution = (args['output_height'], args['output_width'])

    self.video_out = cv2.VideoWriter(output_file, fourcc, 12.0, resolution)

  def __del__(self):
    # Release everything if job is finished
    self.video_in.release()
    if hasattr(self, 'video_out'):
      self.video_out.release()
    cv2.destroyAllWindows()

  def start(self):
    self.proc = Process(target=self._run, args=())
    self._start_time = datetime.datetime.now()
    self._end_time = None
    self._frame_count = 0
    self._stopped = False
    self.proc.start()
    return self

  def _run(self):
    while True:
      if self._stopped:
        break
      ret, frame = self.video_in.read()
      if ret:
        self._last_frame_idx = int(self.video_in.get(cv2.CAP_PROP_POS_FRAMES))

        # Reverse the frame along the Y-axis so it's like
        # looking in a mirror.
        self._last_frame = cv2.flip(frame,1)
        self._frame_count += 1

        if self.output_file:
          cv2.imwrite(output_file, self._last_frame)

        if self.display:
          cv2.imshow('frame', self._last_frame)

      else:
        raise RuntimeException("Video input disconnected.")
    self._end_time = datetime.datetime.now()

  def stop(self):
    self._stopped = True

  @property
  def last_frame_and_index(self):
    # Get the last frame that was read and its frame index.
    return self._last_frame, self._last_frame_idx

  @property
  def theoretical_fps(self):
    return int(self.video_in.get(cv2.CAP_PROP_FPS))

  @property
  def actual_fps(self):
    return self._frame_count / self.time_elapsed

  @property
  def width(self):
    return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
  
  @property
  def height(self):
    return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

  @property
  def time_elapsed(self):
    if self._stopped:
      return (self._end_time - self._start_time).total_seconds()
    return (datetime.datetime.now() - self._start_time).total_seconds()
