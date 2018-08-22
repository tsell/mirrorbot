"""
Base class for feature detectors.
"""

import matplotlib.pyplot as plt
import cv2

import logging

class BaseDetector(object):
  
  def __init__(self):
    logging.info('New detector: %s' % (self.__name__))

  def getFeatures(self, cvImg):
    """
    Returns the array of features in an image.

    Args:
      cvImg: An image in cv2's standard img format.
    """
    logging.info('Computing keypoints')
    raise NotImplementedError()

  def frameProcessor(self, frame):
    return cv2.drawKeypoints(frame, self.getFeatures(frame), outImage=None, color=(0,255,0), flags=0)

  def drawFeatures(self, cvImg):
    """
    Displays the image with features highlighted.
    """
    plt.imshow(self.frameProcessor(cvImg))
    plt.show()
    logging.info('Displayed keypoints.')
    cv2.waitKey(0)
