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

  def captureFrame(self, output_file=None):
    """Capture a single frame of video and write it to disk.
    
    Args:
      image_path: if set, write the frame to this file.
    """
    # Read in a single frame.
    ret, frame = self.video_in.read()

    if ret==True:
      # Reverse the frame along the Y-axis so it's like
      # looking in a mirror.
      frame = cv2.flip(frame,1)

      # Write the frame.
      if output_file:
          cv2.imwrite(output_file, frame)

      # Display the frame.
      cv2.imshow('frame',frame)
    else:
      raise RuntimeException("Video input disconnected.")
 
    return frame

  def captureVideo(self,
     output_file='output.avi',
     resolution=(640,480)):
    """Capture the video stream until it closes.
    
    Args:
      output_file: if set, write the video output to this file.
      resolution: resolution to save the video output at.
    """

    # Define the codec and the output writer.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    self.video_out = cv2.VideoWriter(output_file, fourcc, 20.0, resolution)

    while(self.video_in.isOpened()):
      frame = self.captureFrame()
      self.video_out.write(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def __del__(self):
    # Release everything if job is finished
    self.video_in.release()
    self.video_out.release()
    cv2.destroyAllWindows()
