import gc
import os
import time

import pandas as pd
import numpy as np

from pyts.classification import KNeighborsClassifier, BOSSVS

from sklearn.model_selection import StratifiedShuffleSplit

def load_samples_into_ram(sample_names, data_dir, data_size):
  # TRICKY: Preallocate based on how much we will need. Makes memory usage and performance better    
  start = time.process_time()
  prealloc = np.zeros((sample_names.shape[0], data_size), dtype="float32")    
  for i in range(0, len(sample_names)):
    if i % 10000 == 0:
      print(f"Loading sample {i}")
      gc.collect()
    sample_path = os.path.join(data_dir, f"{sample_names.iloc[i]}.npy")
    prealloc[i, :] = np.load(sample_path)
  gc.collect()
  end = time.process_time()
  elapsed_sec = (end-start)
  print(f"Loaded {prealloc.shape[0]} samples in {elapsed_sec:.2f} seconds")
  return prealloc

def load_dataset(data_dir, data_size, n_splits=1, test_size=0.2):
  dataset_meta_path = os.path.join(data_dir, "labels.pkl")
  dataset_meta = pd.read_pickle(dataset_meta_path)

  samples = load_samples_into_ram(dataset_meta["img"], data_dir, data_size)
  species = dataset_meta["species"].to_numpy()
  breed = dataset_meta["breed"].to_numpy()

  sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
  for train_index, test_index in sss.split(samples, species):
    train_samples_species, test_samples_species = samples[train_index], samples[test_index]
    train_labels_species,  test_labels_species  = species[train_index], species[test_index]

  for train_index, test_index in sss.split(samples, breed):
    train_samples_breed, test_samples_breed = samples[train_index], samples[test_index]
    train_labels_breed,  test_labels_breed  = breed[train_index], breed[test_index]

  return {
    # Species
    "train_samples_species": train_samples_species,
    "train_labels_species": train_labels_species,
    "test_samples_species": test_samples_species,
    "test_labels_species": test_labels_species,
    # Breed
    "train_samples_breed": train_samples_breed,
    "train_labels_breed": train_labels_breed,
    "test_samples_breed": test_samples_breed,
    "test_labels_breed": test_labels_breed,
  }

def report_metrics(metrics):
  print("Metrics for this run:")
  for k, v in metrics.items():
    print(f"\t{k}: {v}")

def run_model(train_x, train_y, test_x, test_y, model):
  start_train = time.process_time()
  try:
    model.fit(train_x, train_y)
  except Exception as e:
    print(f"Encountered an exception fitting with {model}: {e}")
    del model
    gc.collect()
    return None
    
  # Metrics
  end_train = time.process_time()
  train_time_sec = (end_train-start_train)
  start_test = time.process_time()
  accuracy = model.score(test_x, test_y)
  end_test = time.process_time()
  test_time_sec = (end_test-start_test)
  # TODO: Discover other metrics here

  del model
  gc.collect()
  metrics = {
    "train_time_sec": train_time_sec,
    "test_time_sec": test_time_sec,
    "accuracy": accuracy
  }
  report_metrics(metrics)
  return metrics

def run_knn(train_x, train_y, test_x, test_y, args):
  knn = KNeighborsClassifier(**args)
  return run_model(train_x, train_y, test_x, test_y, knn)
    
def run_bossvs(train_x, train_y, test_x, test_y, args):
  boss = BOSSVS(**args)
  return run_model(train_x, train_y, test_x, test_y, boss)

def run_classifiers_on_dataset(dataset):
  # TODO: Maybe do some optimization of these params
  knn_args = { "metric": "euclidean"}
  bossvs_args = { "word_size": 2, "window_size": 16 }

  print("Starting run: Species KNN")
  knn_species_metrics = run_knn(
    dataset["train_samples_species"],
    dataset["train_labels_species"],
    dataset["test_samples_species"],
    dataset["test_labels_species"],
    knn_args
  )

  print("Starting run: Breed KNN")
  knn_breed_metrics = run_knn(
    dataset["train_samples_breed"],
    dataset["train_labels_breed"],
    dataset["test_samples_breed"],
    dataset["test_labels_breed"],
    knn_args
  )
  
  print("Starting run: Species BOSSVS")
  boss_species_metrics = run_bossvs(
    dataset["train_samples_species"],
    dataset["train_labels_species"],
    dataset["test_samples_species"],
    dataset["test_labels_species"],
    bossvs_args
  )

  print("Starting run: Breed BOSSVS")
  boss_breed_metrics = run_bossvs(
    dataset["train_samples_breed"],
    dataset["train_labels_breed"],
    dataset["test_samples_breed"],
    dataset["test_labels_breed"],
    bossvs_args
  )
  
  return {
    "knn_species_metrics": knn_species_metrics,
    "knn_breed_metrics": knn_breed_metrics,
    "boss_species_metrics": boss_species_metrics,
    "boss_breed_metrics": boss_breed_metrics,
  }