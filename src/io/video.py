import numpy as np
import cv2

class Video(object):
  def __init__(self,
    source=0,
    output_file='output.avi',
    resolution=(640,480)):

    # Connect to the webcam/input device.
    self.video_in = cv2.VideoCapture(source)

    # Define the codec and the output writer.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    self.video_out = cv2.VideoWriter(output_file, fourcc, 20.0, resolution)

  def captureFrame(self):
    # Read in a single frame.
    ret, frame = self.video_in.read()

    if ret==True:
      # Reverse the frame so it's like looking in a mirror.
      frame = cv2.flip(frame,0)

      # Write the flipped frame.
      self.video_out.write(frame)

      # Display the frame.
      cv2.imshow('frame',frame)
    else:
      raise RuntimeException("Video input disconnected.")

  def captureVideo(self):
    while(self.video_in.isOpened()):
      self.captureFrame()
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def __del__(self):
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
