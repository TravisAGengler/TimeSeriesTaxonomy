#!/usr/bin/env python3

import numpy as np

from skimage.color import rgb2gray

def a1(img_data):
  grayscale = rgb2gray(img_data)
  a1_data = np.empty(grayscale.shape[0] * grayscale.shape[1])
  for r in range(grayscale.shape[0]):
    o = grayscale.shape[1]*r
    for c in range(grayscale.shape[1]):
      a1_data[o + c] = grayscale[r,c]
  return a1_data