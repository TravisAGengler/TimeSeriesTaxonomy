#!/usr/bin/env python3

import numpy as np

from skimage import exposure, filters, color, io

from skimage import data
from matplotlib import pyplot as plt

def adaptive_equalization(img_data):
  img_hsv = color.convert_colorspace(img_data, "rgb", "hsv")
  img_adapteq = exposure.equalize_adapthist(img_hsv, clip_limit=0.02)
  img_eq_rgb = color.convert_colorspace(img_adapteq, "hsv", "rgb")
  return img_eq_rgb

def unsharp_masking(img_data, radius=3, amount=1):
  return filters.unsharp_mask(img_data, radius, amount)

def to_histogram(img_data, bins = 64):
  # TODO: Should we weight centralized pixels more than others? 64x64 Gausian centered? This is the place to do it
  h = np.histogram(img_data, bins)[0]
  return h / h.sum()

def a2(img_data):
  img_data_no_alpha = img_data[:,:,:3]
  unsharp = unsharp_masking(img_data_no_alpha)
  adapt = adaptive_equalization(unsharp)
  r = to_histogram(adapt[:,:,0])
  g = to_histogram(adapt[:,:,1])
  b = to_histogram(adapt[:,:,2])
  return r, g, b