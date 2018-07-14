#!/usr/bin/env python

from video import video
from features import sift

import argparse
import cv2
import numpy as np

def monoChromeFrameProcessor(cvImg):
  thresh = 127
  im_bw = cv2.threshold(cvImg, thresh, 255, cv2.THRESH_BINARY)[1]
  return im_bw

def main():
  v = video.VideoStream()
  v.captureVideo(frame_processor=sift.SIFT().frameProcessor)
  del v

if __name__=="__main__":
  main()
