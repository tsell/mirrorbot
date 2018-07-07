"""
SIFT feature detection, see
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
and
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
"""

from features.base_detector import BaseDetector

import cv2
import numpy as np

class SIFT(BaseDetector):

  def __init__(self):
    self.sift = cv2.xfeatures2d.SIFT_create()

  def getFeatures(self, cvImg):
    gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    key_points = self.sift.detect(gray, None)
    return key_points
