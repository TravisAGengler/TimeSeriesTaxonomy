#!/usr/bin/env python3

import argparse
import uuid
import os
import csv

import pandas as pd
import numpy as np

from typing import List, Tuple
from enum import Enum
from xml.etree import ElementTree
from skimage import io, img_as_ubyte
from skimage.transform import resize

from data_types import Path, Label, Species, Breed, BoundingBox, OrigDataset, Dataset

class Command(Enum):
  preprocess = 1
  a1 = 2
  a2 = 3
  a3 = 4

  def __str__(self):
    return self.name

  @staticmethod
  def from_string(s):
    try:
      return Command[s]
    except KeyError:
      raise ValueError()

def label_to_breed(label: Label) -> Breed:
  try:
    breed = Breed[label]
    return breed
  except KeyError:
    # These are the special cases
    if label == "laperm":
      return Breed.la_perm
  raise "No breed found"

def get_bounding_boxes_from_xml(xml_path: Path) -> BoundingBox:
  xml_tree = ElementTree.parse(xml_path)
  root = xml_tree.getroot()
  bbs = []
  for bb_xml in root.findall(".//bndbox"):
    x = int(bb_xml.find("xmin").text)
    y = int(bb_xml.find("xmin").text)
    x_max = int(bb_xml.find("xmax").text)
    y_max = int(bb_xml.find("ymax").text)
    bbs.append(BoundingBox(x, y, x_max-x, y_max-y))
  return bbs

def preprocess_img_cat(img_path, bb: BoundingBox=None):
  img_data = io.imread(img_path)
  img_w = img_data.shape[1] // 6
  img_h = img_data.shape[0] // 6
  img_data = img_data[img_h:img_data.shape[0]-img_h, img_w:img_data.shape[1]-img_w]
  return img_as_ubyte(resize(img_data, (64,64), anti_aliasing=True))

def preprocess_img_dog(img_path, bb: BoundingBox):
  img_data = io.imread(img_path)
  img_data = img_data[bb.y:bb.y+bb.h, bb.x:bb.x+bb.w]
  return img_as_ubyte(resize(img_data, (64,64), anti_aliasing=True))

def preprocess_cat_img(img_path):
  # Identify the first 1/6 and last 1/6 of the image on both x and y axis
  # Take the largest window axis so that we produce a square. Crop to square
  # Then, resize image to 64x64

  # img_data = io.imread(img_path)
  # h, w, d = img_data.shape
  # x_u = w / 6
  # y_u = h / 6
  # x_w = x_u * 4
  # y_w = y_u * 4
  # e_w = min([x_w, y_w])
  # crop_img = img_data[y1:y2, x1:x2]
  # resize_img = resize(crop_img, (64,64), anti_aliasing=True)
  # return resize_img
  pass

def preprocess_dog_img(img_path, bounding_box):
  # Since the dog images are ALREADY segmented with a bounding box, this process is different
  # Make area a square
  # Resize to 64x64
  pass

def read_cat_orig(in_dir: Path) -> List[Tuple[Path, Label]]:
  cat_samples = []
  cat_imgs_dir = os.path.join(in_dir, "images")
  for breed_dir in os.listdir(cat_imgs_dir):
    label = breed_dir.lower().replace(" - ", "-").replace(" ", "_").replace("-", "_")
    cat_img_dir = os.path.join(cat_imgs_dir, breed_dir)
    for cat_img in os.listdir(cat_img_dir):
      cat_img_path = os.path.abspath(os.path.join(cat_img_dir, cat_img))
      cat_sample = (cat_img_path, label_to_breed(label))
      cat_samples.append(cat_sample)
  return cat_samples

def read_dog_orig(in_dir: Path) -> List[Tuple[Path, BoundingBox, Label]]:
  dog_samples = []
  dog_imgs_dir = os.path.join(in_dir, "images/Images")
  dog_bound_boxes_dir = os.path.join(in_dir, "annotations/Annotation")
  for breed_dir in os.listdir(dog_imgs_dir):
    label = breed_dir.lower().split('-', 1)[1].replace(" - ", "-").replace(" ", "_").replace("-", "_")
    dog_img_dir = os.path.join(dog_imgs_dir, breed_dir)
    bound_box_dir = os.path.join(dog_bound_boxes_dir, breed_dir)
    for dog_img in os.listdir(dog_img_dir):
      dog_img_path = os.path.abspath(os.path.join(dog_img_dir,dog_img))
      dog_bounding_box_path = os.path.join(bound_box_dir, os.path.splitext(dog_img)[0])
      for bounding_box in get_bounding_boxes_from_xml(dog_bounding_box_path):
        dog_sample = (dog_img_path, bounding_box, label_to_breed(label))
        dog_samples.append(dog_sample)
  return dog_samples

def read_orig(in_dir: Path) -> OrigDataset:
  print(f"Reading original data from '{in_dir}'")

  cat_in_dir = os.path.join(in_dir, "cat_breeds")
  dog_in_dir = os.path.join(in_dir, "dog_breeds")

  cat_samples = read_cat_orig(cat_in_dir)
  dog_samples = read_dog_orig(dog_in_dir)

  return OrigDataset(cat_samples, dog_samples)

def read_processed(in_dir: Path) -> Dataset:
  pass

def preprocess_orig(orig_data: OrigDataset, out_dir: Path) -> Dataset:
  samples = []

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # Cats
  n_cat = 0
  for sample in orig_data.cat_breeds:
    img_path = sample[0]
    l0 = Species.cat
    l1 = sample[1]
    try:
      print(f"Preprocessing cat {os.path.basename(img_path)}")
      proc_img = preprocess_img_cat(img_path)
      new_filename = str(uuid.uuid4())
      out_path = os.path.join(out_dir, new_filename + ".png")
      io.imsave(out_path, proc_img)
      samples.append({"img" : new_filename, "species" : l0, "breed" : l1})
    except Exception as e:
      print(f"... Failed to preprocess {img_path}: {str(e)}")
      continue
    n_cat += 1

  # Dogs
  n_dog = 0
  for sample in orig_data.dog_breeds:
    img_path = sample[0]
    bound = sample[1]
    l0 = Species.dog
    l1 = sample[2]
    try:
      print(f"Preprocessing dog {os.path.basename(img_path)}")
      proc_img = preprocess_img_dog(img_path, bb=bound)
      new_filename = str(uuid.uuid4())
      out_path = os.path.join(out_dir, new_filename + ".png")
      io.imsave(out_path, proc_img)
      samples.append({"img" : new_filename, "species" : l0, "breed" : l1})
    except Exception as e:
      print(f"... Failed to preprocess {img_path}: {str(e)}")
      continue
    n_dog += 1

  return pd.DataFrame(samples), n_cat, n_dog

def data_preprocess(in_dir, out_dir):
  print(f"Performing data_preprocess('{in_dir}','{out_dir}')")
  orig_data = read_orig(in_dir)
  print(f"Found {len(orig_data.cat_breeds)} cat samples")
  print(f"Found {len(orig_data.dog_breeds)} dog samples")
  new_data, n_cat, n_dog = preprocess_orig(orig_data, out_dir)
  dataset_out_path = os.path.join(out_dir, "labels.pkl")
  new_data.to_pickle(dataset_out_path)
  print(f"Finished preprocessing {n_cat} cat samples and {n_dog} dog samples, labels written to {dataset_out_path}")

def data_a1(in_dir: Path, out_dir: Path):
  print(f"Performing data_a1('{in_dir}','{out_dir}')")

def data_a2(in_dir: Path, out_dir: Path):
  print(f"Performing data_a2('{in_dir}','{out_dir}')")

def data_a3(in_dir: Path, out_dir: Path):
  print(f"Performing data_a3('{in_dir}','{out_dir}')")

def main():
  command, in_dir, out_dir = parse_args()

  if command == Command.preprocess:
    data_preprocess(in_dir, out_dir)
  elif command == Command.a1:
    data_a1(in_dir, out_dir)
  elif command == Command.a2:
    data_a2(in_dir, out_dir)
  elif command == Command.a3:
    data_a3(in_dir, out_dir)

def parse_args():
  parser = argparse.ArgumentParser(prog="data", description="")
  parser.add_argument("-c", "--command", type=Command.from_string, required=True,
                  help=f"The command to run.", choices=list(Command))
  parser.add_argument("-i", "--in_dir", metavar="PATH", type=str, required=True,
                  help="The original data directory.")
  parser.add_argument("-o", "--out_dir", metavar="PATH", type=str, required=True,
                  help="The directory where processed data will be placed")
  args = vars(parser.parse_args())
  return args["command"], args["in_dir"],  args["out_dir"]

if __name__ == "__main__":
  main()