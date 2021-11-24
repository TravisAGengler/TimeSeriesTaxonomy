#!/usr/bin/env python3

import numpy as np

from scipy import ndimage
from skimage import exposure, filters, color, io
from sklearn.preprocessing import minmax_scale

from a2 import adaptive_equalization

mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 1, 2, 2, 2, 2, 1, 0],
                 [0, 1, 2, 2, 2, 2, 1, 0],
                 [0, 1, 2, 2, 2, 2, 1, 0],
                 [0, 1, 2, 2, 2, 2, 1, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

def highpass_gauss_filter(img_data):
  lowpass = ndimage.gaussian_filter(img_data, 3)
  gauss_highpass = img_data - lowpass
  return gauss_highpass

def get_window(img_data, x, y, size):
  return img_data[y:y+size, x:x+size]

def get_window_val(img_data, x, y, mask_bit):
  if mask_bit == 0:
    vals = np.zeros(1)
    size = 8
  elif mask_bit == 1:
    vals = np.zeros(4)
    size = 4
  elif mask_bit == 2:
    vals = np.zeros(16)
    size = 2
  step = 8//size
  for x_s in range(0,step):
    for y_s in range(0,step):
      window = get_window(img_data, x + x_s, y + y_s, size)
      vals[y_s*step+x_s] = np.mean(window)
  return vals

def center_focus(img_data):
  offset_mask_2 = 0
  offset_mask_1 = 0
  offset_mask_0 = 0
  center_focus_data_mask_2 = np.zeros(16*16)
  center_focus_data_mask_1 = np.zeros(20*4)
  center_focus_data_mask_0 = np.zeros(28*1)
  for x in range(0, mask.shape[0]):
    for y in range(0, mask.shape[1]):
      mask_bit = mask[y,x]
      window_val = get_window_val(img_data, x, y, mask_bit)
      if mask_bit == 0:
        center_focus_data_mask_0[offset_mask_0: offset_mask_0 + 1] = window_val
        offset_mask_0 += 1
      elif mask_bit == 1:
        center_focus_data_mask_1[offset_mask_1: offset_mask_1 + 4] = window_val
        offset_mask_1 += 4
      elif mask_bit == 2:
        center_focus_data_mask_2[offset_mask_2: offset_mask_2 + 16] = window_val
        offset_mask_2 += 16

  return np.concatenate((center_focus_data_mask_2, center_focus_data_mask_1, center_focus_data_mask_0))

def a3(img_data):
  img_data_no_alpha = img_data[:,:,:3]
  adapt = adaptive_equalization(img_data_no_alpha)
  r = center_focus(highpass_gauss_filter(adapt[:,:,0]))
  g = center_focus(highpass_gauss_filter(adapt[:,:,1]))
  b = center_focus(highpass_gauss_filter(adapt[:,:,2]))
  return minmax_scale(np.concatenate((r.astype("float32"), g.astype("float32"), b.astype("float32"))), feature_range=(0,1))

