#!/usr/bin/env python

from features import sift

import argparse
import cv2
import numpy as np

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# Get the image.
img = cv2.imread(args['image'])
img = cv2.resize(img, (600,600), interpolation=cv2.INTER_CUBIC)
print('Computing keypoints')
kp = sift.SIFT().drawFeatures(img)
print('Done!')
