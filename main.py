#!/usr/bin/env python3

import gc
import os

import pandas as pd

from utils import load_dataset, run_classifiers_on_dataset
from skimage import io

from data import preprocess_img_cat
from matplotlib import pyplot as plt

from a1 import a1
from a3 import a3
from a2 import a2, unsharp_masking, adaptive_equalization

def test_a3():
  img_path = "data/preprocessed/0a0fa824-668a-4051-90be-8020a6a83cbb.png"
  img_data = io.imread(img_path)
  a3_data = a3(img_data)
  print(a3_data)
  print(a3_data.max())
  print(a3_data.min())

def report_dataset_stats():
  data_dir = "data/a2"
  dataset_meta_path = os.path.join(data_dir, "labels.pkl")
  dataset_meta = pd.read_pickle(dataset_meta_path)
  print(dataset_meta)

def get_cat_example():
  path_cat = "data/orig/cat_breeds/images/Abyssinian/8225343_254.jpg"
  cat_orig = io.imread(path_cat)
  cat_img = preprocess_img_cat(path_cat)
  # img_data_no_alpha = cat_img[:,:,:3]
  # unsharp = unsharp_masking(img_data_no_alpha)
  adapt = adaptive_equalization(cat_img)
  # io.imshow(cat_img)
  # plt.show()
  # a2_cat = a2(cat_img)
  # print(a2_cat)
  io.imsave("a3_adapt.png", adapt)
  # io.imsave("preprocess_cat.png", cat_img)


def test_run_a2():
  dataset_a2_sample_size = 64*3
  dataset_a2 = load_dataset("data/a2", dataset_a2_sample_size)
  dataset_a2_metrics = run_classifiers_on_dataset(dataset_a2)
  del dataset_a2
  gc.collect()

def test_run_a3():
  dataset_a3_sample_size = ((16 * 16) + (20 * 4) + (28 * 1)) * 3
  dataset_a3 = load_dataset("data/a3", dataset_a3_sample_size)
  dataset_a3_metrics = run_classifiers_on_dataset(dataset_a3)
  del dataset_a2
  gc.collect()

def main():
  get_cat_example()

if __name__ == "__main__":
  main()