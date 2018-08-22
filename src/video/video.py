"""
Video in/out utility.
"""

import cv2
import numpy as np

class VideoStream(object):
  """Represents a video being streamed in from a camera."""

  def __init__(self, source=0):

    # Connect to the webcam/input device.
    self.video_in = cv2.VideoCapture(source)

  def captureFrame(self, output_file=None, display=True, frame_processor=None):
    """Capture a single frame of video and write it to disk.
    
    Args:
      image_path: if set, write the frame to this file.
      display: if True, display the captured frame in X.
      frame_processor: if set, call frame_processor(frame, frame_idx) on the frame.
    Returns: the frame captured, as a numpy matrix.
    """
    # Read in a single frame.
    ret, frame = self.video_in.read()
    if ret:
      frame_idx = int(self.video_in.get(cv2.CAP_PROP_POS_FRAMES))
      if frame_processor is not None:
          frame = frame_processor(frame, frame_idx)

      # Reverse the frame along the Y-axis so it's like
      # looking in a mirror.
      frame = cv2.flip(frame,1)

      # Write the frame.
      if output_file:
        cv2.imwrite(output_file, frame)

      if display:
        cv2.imshow('frame', frame)
    else:
      raise RuntimeException("Video input disconnected.")
 
    return frame, frame_idx

  def captureVideo(self,
     output_file='output.avi',
     resolution=(640,480),
     frame_processor=None):
    """Capture the video stream until it closes.
    
    Args:
      output_file: if set, write the video output to this file.
      resolution: resolution to save the video output at.
      frame_processor: a function(frame, frame_idx) which processes frames.
    """

    # Define the codec and the output writer.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    self.video_out = cv2.VideoWriter(output_file, fourcc, 12.0, resolution)

    while(self.video_in.isOpened()):
      frame = self.captureFrame(frame_processor=frame_processor)
      frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_CUBIC)
      self.video_out.write(frame)
      # Press q to stop capturing.
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def __del__(self):
    # Release everything if job is finished
    self.video_in.release()
    if hasattr(self, 'video_out'):
      self.video_out.release()
    cv2.destroyAllWindows()
