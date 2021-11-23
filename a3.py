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

def lowpass_gauss_filter(img_data, radius=3, amount=1):
  # TODO: Implement this
  # https://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
  return img_data

def to_histogram(img_data, bins = 64):
  # TODO: Should we weight centralized pixels more than others? 64x64 Gausian centered? This is the place to do it
  h = np.histogram(img_data, bins)[0]
  return h / h.sum()

def a3(img_data):
  img_data_no_alpha = img_data[:,:,:3]
  adapt = adaptive_equalization(img_data_no_alpha)
  lowpass = lowpass_gauss_filter(adapt)
  r = to_histogram(lowpass[:,:,0])
  g = to_histogram(lowpass[:,:,1])
  b = to_histogram(lowpass[:,:,2])
  return np.concatenate((r.astype("float32"), g.astype("float32"), b.astype("float32")))

