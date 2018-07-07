"""
Tests for SIFT feature DETECTOR.
"""

from io.video import VideoStream
from features import sift

import cv2
import numpy as np

import copy
import unittest

VID = VideoStream()
IMG = VID.captureFrame(display=False)
DETECTOR = sift.SIFT()

class SiftTest(unittest.TestCase):

  def testSameImageSameKeypoints(self):
    img_copy = copy.deepcopy(IMG)
    kp1 = DETECTOR.getFeatures(IMG)
    kp2 = DETECTOR.getFeatures(img_copy)
    assert kp1 == kp2

  def testKeypointsNotEmpty(self)
    kp = DETECTOR.getFeatures(IMG)
    assert np.any(kp)

if __name__=='__main__':
  unittest.main()
