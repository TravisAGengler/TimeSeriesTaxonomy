#!/usr/bin/env python3

import gc
import os
import time

import pandas as pd
import numpy as np

from pyts.classification import KNeighborsClassifier, BOSSVS

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def load_samples_into_ram(sample_names, data_dir, data_size):
  # TRICKY: Preallocate based on how much we will need. Makes memory usage and performance better    
  start = time.perf_counter_ns()
  prealloc = np.zeros((sample_names.shape[0], data_size), dtype="float32")    
  for i in range(0, len(sample_names)):
    if i % 10000 == 0:
      print(f"Loading sample {i}")
      gc.collect()
    sample_path = os.path.join(data_dir, f"{sample_names.iloc[i]}.npy")
    prealloc[i, :] = np.load(sample_path)
  gc.collect()
  end = time.perf_counter_ns()
  elapsed_sec = (end-start) / 1e9
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

def get_roc(test_y, test_prob_scores, is_multiclass):
  if not is_multiclass:
    # For two class predictions:
    # col 0 for predicting class "dog" (First col), Val 1. Cat is 0
    roc_false_pos_rate, roc_true_pos_rate, _ = roc_curve(test_y, test_prob_scores[:, 0], pos_label=1)
    roc_auc = auc(roc_false_pos_rate, roc_true_pos_rate) 
    return roc_false_pos_rate, roc_true_pos_rate, roc_auc
  return None, None, None # We would need a ROC curve for every breed... Thats not feasible

def get_roc_knn(model_knn, test_x, test_y):
  test_prob_scores = model_knn._clf.predict_proba(test_x)
  return get_roc(test_y, test_prob_scores, len(set(test_y)) > 2)

def get_roc_boss(model_boss, test_x, test_y):
  # https://github.com/alan-turing-institute/sktime/blob/v0.8.1/sktime/classification/base.py#L244
  # 1 on predicted class, 0 everywhere else
  n_classes = len(set(test_y))
  # We need to short circut here if we have more than 2 classes
  if n_classes > 2:
    return None, None, None
  test_preds = model_boss.predict(test_x)
  test_prob_scores = np.zeros((test_x.shape[0], n_classes))
  for i in range(0, test_x.shape[0]):
      test_prob_scores[i, test_preds[i]-1] = 1
  return get_roc(test_y, test_prob_scores, n_classes > 2)

def evaluate_model(train_x, train_y, test_x, test_y, start_train, model):
  end_train = time.perf_counter_ns()
  train_time_sec = (end_train-start_train) / 1e9

  start_test = time.perf_counter_ns()

  train_pred_y = model.predict(train_x)
  test_pred_y = model.predict(test_x)

  end_test = time.perf_counter_ns()
  test_time_sec = (end_test-start_test) / 1e9

  train_confusion = confusion_matrix(train_y, train_pred_y)
  test_confusion = confusion_matrix(test_y, test_pred_y)
  train_accuracy = accuracy_score(train_y, train_pred_y)
  train_precision = np.nanmean(np.diag(train_confusion) / np.sum(train_confusion, axis = 0))
  train_recall = np.nanmean(np.diag(train_confusion) / np.sum(train_confusion, axis = 1))
  test_accuracy = accuracy_score(test_y, test_pred_y)
  test_precision = np.nanmean(np.diag(test_confusion) / np.sum(test_confusion, axis = 0))
  test_recall = np.nanmean(np.diag(test_confusion) / np.sum(test_confusion, axis = 1))

  get_roc = get_roc_knn if hasattr(model, '_clf') else get_roc_boss
  roc_false_pos_rate, roc_true_pos_rate, roc_auc = get_roc(model, test_x, test_y)

  metrics = {
    "train_time_sec": train_time_sec,
    "test_time_sec": test_time_sec,
    "train_accuracy": train_accuracy,
    "train_precision": train_accuracy,
    "train_recall": train_accuracy,
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "roc_false_pos_rate": roc_false_pos_rate,
    "roc_true_pos_rate": roc_true_pos_rate,
    "roc_auc": roc_auc,
  }
  return metrics

def run_model(train_x, train_y, test_x, test_y, model):
  # Train
  start_train = time.perf_counter_ns()
  try:
    model.fit(train_x, train_y)
    metrics = evaluate_model(train_x, train_y, test_x, test_y, start_train, model)
    report_metrics(metrics)
  except Exception as e:
    print(f"Encountered an exception running model {model}: {e}")
    metrics = None

  # Cleanup and return metrics
  del model
  gc.collect()
  return metrics

def run_knn(train_x, train_y, test_x, test_y, args):
  knn = KNeighborsClassifier(**args)
  return run_model(train_x, train_y, test_x, test_y, knn)

def run_bossvs(train_x, train_y, test_x, test_y, args):
  boss = BOSSVS(**args)
  return run_model(train_x, train_y, test_x, test_y, boss)

def run_classifiers_on_dataset(dataset):
  # TODO: Maybe do some optimization of these params
  knn_args = { "metric": "euclidean", "n_neighbors" : 1 }
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