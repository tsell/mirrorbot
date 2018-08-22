#!/usr/bin/env python

import queue
from threading import Thread


class FrameProcessorThread(Thread):
  """A thread which processes frames from one queue
  and puts them into another queue."""
  def __init__(self,
      frame_processor_func,
      input_queue):
    super(FrameProcessorThread, self).__init__()
    self.input_queue = input_queue
    self.output_queue = queue.Queue()
    self.frame_processor_func = frame_processor_func

  def run(self):
    """Process frames from input_queue and put them in
    output_queue, as tuples of (frame_idx, frame).
    
    Join with parent thread when queue is empty.
    """
    while True:
      try:
        # Wait for the queue to be nonempty.
        frame_idx, input_frame = self.input_queue.get(block=True, timeout=60)
      except queue.Empty as e:
        print("Input queue was empty.")
        break;
      output_frame = self.frame_processor_func(frame)
      if output_frame:
        self.output_queue.put((frame_idx, output_frame))
      self.input_queue.task_done()


class BufferedVideoProcessor(object):
  """A multithreaded video processor that runs a series of
  functions on each frame from video_stream."""

  def __init__(self, video_stream, functions, buffer_capacity = 12):
    self._video_stream = video_stream
    self._input_queue = queue.Queue(buffer_capacity)

    # Build a chain of frame processors.
    last_queue = self._input_queue
    self._threads = []
    for func in functions:
      thread = FrameProcessorThread(func, last_queue)
      self._threads.append(thread)
      last_queue = thread.output_queue

    self._output_queue = last_queue
  
  def process(self,
      frame_processor_func = lambda frame: frame,
      max_frames = None):
    """Read frames, store in input queue, and run
    callback() on each frame in separate threads."""

    frames_processed = 0
    while not max_frames or frames_processed < max_frames:
      if not self._input_queue.full():
        frame, frame_idx = self._video_stream.captureFrame(display=False)
        self._input_queue.put((frame_idx, frame))

    self._output_queue.join()

