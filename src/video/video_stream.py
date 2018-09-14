"""
Video in/out utility.
"""
import ctypes
import datetime
import logging as log
import tempfile

from multiprocessing import Process, Value, RLock, Queue

import cv2
import numpy as np

class VideoStream(object):
  def __init__(self, args):
    self._started = False
    lock = RLock()
    # Store start/end time and frame count, for computing actual FPS.
    self._start_time = Value(ctypes.c_double, 0, lock=lock)
    self._end_time = Value(ctypes.c_double, 0, lock=lock)
    self._frame_count = Value(ctypes.c_uint32, 0, lock=lock)

    # Store whether the process should be interrupted.
    self._started = Value(ctypes.c_bool, False, lock=lock)
    self._stopped = Value(ctypes.c_bool, False, lock=lock)
    
    # Define video input from webcam or other input source.
    self.src = args['video_source'] or 0
    
    # Figure out the properties of our webcam.
    video_in = cv2.VideoCapture(self.src)
    self.theoretical_fps =  int(video_in.get(cv2.CAP_PROP_FPS))
    self.width = args['output_width'] or int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height =  args['output_height'] or int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_in.release()
    
    # Should we display the webcam input live?
    self.display = not args['nodisplay']
    
    # Should we record the webcam input?
    self.record = not args['norecord']

    # How many frames?
    self.num_frames = args['num_frames']

    # Define video codec and output writer.
    if self.record:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      resolution = (self.width, self.height)
      _, output_file = tempfile.mkstemp(
          prefix='output', suffix='.avi', dir=args['output_dir'])
      self.video_out = cv2.VideoWriter(output_file, fourcc, 12.0, resolution)
      log.info('Recording enabled to file: %s' % output_file)

    # Keep three frames in the buffer.
    self._frame_queue = Queue(maxsize=args['frame_buffer_size'])

  def __del__(self):
    # Release everything if job is finished
    if hasattr(self, 'proc') and self.proc.is_alive():
      self.proc.terminate()
      self.proc.join()
      if self.record:
        self.video_out.relase()
    cv2.destroyAllWindows()

  def start(self):
    self.proc = Process(target=self._run, daemon=True, kwargs={
      'num_frames': self.num_frames
      })
    self._start_time.value = datetime.datetime.now().timestamp()
    self._end_time.value = 0
    self._frame_count.value = 0
    self._started.value = True
    self._stopped.value = False
    self.proc.start()
    return self

  def _run(self, num_frames):
    video_in = cv2.VideoCapture(self.src)
    log.debug('Started video subprocess.')
    i = 0
    while i < num_frames or num_frames == 0:
      i += 1
      if self._stopped.value:
        break
      ret, frame = video_in.read()
      if ret:
        last_frame_idx = int(video_in.get(cv2.CAP_PROP_POS_FRAMES))

        # Reverse the frame along the Y-axis so it's like
        # looking in a mirror.
        last_frame = cv2.flip(frame,1)
        # Make it the right size, too.
        last_frame = cv2.resize(src=last_frame, dsize=(self.width, self.height))

        log.debug('Waiting for framecount lock.')
        with self._frame_count.get_lock():
          log.debug('Framecount lock acquired.')
          self._frame_count.value += 1

        log.debug('Got frame %d from webcam.' % self._frame_count.value)
        self._frame_queue.put((self._frame_count.value, last_frame))
        log.debug('Put frame %d in queue.' % self._frame_count.value)

      else:
        self._end_time = datetime.datetime.now().timestamp()
        video_in.release()
        raise RuntimeError("Video input disconnected.")
    self._stopped.value = True
    self._end_time = datetime.datetime.now().timestamp()
    video_in.release()

  def stop(self):
    self.assert_started()
    self._stopped.value = True

  @property
  def stopped(self):
    return self._stopped.value

  def assert_started(self):
    assert self._started.value

  @property
  def frame_count(self):
    return self._frame_count.value

  def get_last_index_and_frame(self):
    self.assert_started()
    idx, last_frame = self._frame_queue.get(block=True)
    log.debug('Got frome %d from queue.' % idx)
    if self.display:
      cv2.imshow('frame', last_frame)
    if self.record:
      self.video_out.write(last_frame)
    return idx, last_frame

  @property
  def actual_fps(self):
    self.assert_started()
    return self.frame_count / self.time_elapsed

  @property
  def time_elapsed(self):
    self.assert_started()
    start_time = datetime.datetime.fromtimestamp(self._start_time.value)
    if self._stopped:
      end_time = datetime.datetime.fromtimestamp(self._end_time.value)
      return (end_time - start_time).total_seconds()
    return (datetime.datetime.now() - start_time).total_seconds()
