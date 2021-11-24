#!/usr/bin/env python3

import gc

from utils import load_dataset, run_classifiers_on_dataset
from skimage import io

from a3 import a3

def test_a3():
  img_path = "data/preprocessed/0a0fa824-668a-4051-90be-8020a6a83cbb.png"
  img_data = io.imread(img_path)
  a3_data = a3(img_data)
  print(a3_data)
  print(a3_data.max())
  print(a3_data.min())

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
  test_run_a2()

if __name__ == "__main__":
  main()