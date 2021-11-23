#!/usr/bin/env python3

import gc

from utils import load_dataset, run_classifiers_on_dataset

def main():
  dataset_a2_sample_size = 64*3
  dataset_a2 = load_dataset("data/a2", dataset_a2_sample_size)
  dataset_a2_metrics = run_classifiers_on_dataset(dataset_a2)
  del dataset_a2
  gc.collect()

if __name__ == "__main__":
  main()