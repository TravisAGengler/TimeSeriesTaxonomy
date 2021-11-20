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
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize

from data_types import Path, Label, Species, Breed, BoundingBox, OrigDataset, Dataset

from a1 import a1
from a2 import a2 

LABELS_FILENAME = "labels.pkl"

BREED_COUNTS_TOTAL = {
  # Dogs
  Breed.samoyed : 232,
  Breed.entlebucher : 230,
  Breed.irish_wolfhound : 230,
  Breed.great_pyrenees : 229,
  Breed.shih_tzu : 228,
  Breed.basenji : 227,
  Breed.sealyham_terrier : 227,
  Breed.afghan_hound : 226,
  Breed.leonberg : 226,
  Breed.bernese_mountain_dog : 223,
  Breed.maltese_dog : 222,
  Breed.scottish_deerhound : 222,
  Breed.pug : 220,
  Breed.whippet : 218,
  Breed.saluki : 217,
  Breed.pomeranian : 215,
  Breed.tibetan_terrier : 212,
  Breed.beagle : 208,
  Breed.papillon : 208,
  Breed.siberian_husky : 208,
  Breed.norwich_terrier : 205,
  Breed.airedale : 204,
  Breed.italian_greyhound : 204,
  Breed.african_hunting_dog : 203,
  Breed.chow : 203,
  Breed.lakeland_terrier : 203,
  Breed.australian_terrier : 201,
  Breed.blenheim_spaniel : 201,
  Breed.cairn : 201,
  Breed.norwegian_elkhound : 200,
  Breed.dandie_dinmont : 199,
  Breed.malamute : 199,
  Breed.newfoundland : 198,
  Breed.japanese_spaniel : 195,
  Breed.bloodhound : 194,
  Breed.ibizan_hound : 194,
  Breed.pembroke : 194,
  Breed.lhasa : 191,
  Breed.bedlington_terrier : 190,
  Breed.miniature_pinscher : 190,
  Breed.boston_bull : 189,
  Breed.silky_terrier : 188,
  Breed.english_foxhound : 185,
  Breed.kerry_blue_terrier : 185,
  Breed.labrador_retriever : 185,
  Breed.irish_terrier : 184,
  Breed.basset : 182,
  Breed.west_highland_white_terrier : 182,
  Breed.chesapeake_bay_retriever : 180,
  Breed.standard_schnauzer : 179,
  Breed.norfolk_terrier : 178,
  Breed.rhodesian_ridgeback : 178,
  Breed.saint_bernard : 178,
  Breed.greater_swiss_mountain_dog : 174,
  Breed.bull_mastiff : 173,
  Breed.old_english_sheepdog : 173,
  Breed.bluetick : 172,
  Breed.toy_terrier : 172,
  Breed.border_terrier : 171,
  Breed.keeshond : 171,
  Breed.standard_poodle : 170,
  Breed.american_staffordshire_terrier : 169,
  Breed.english_springer : 167,
  Breed.scotch_terrier : 167,
  Breed.collie : 166,
  Breed.english_setter : 166,
  Breed.yorkshire_terrier : 166,
  Breed.schipperke : 165,
  Breed.borzoi : 164,
  Breed.clumber : 164,
  Breed.giant_schnauzer : 164,
  Breed.great_dane : 164,
  Breed.weimaraner : 164,
  Breed.dingo : 163,
  Breed.wire_haired_fox_terrier : 163,
  Breed.curly_coated_retriever : 162,
  Breed.french_bulldog : 162,
  Breed.mexican_hairless : 162,
  Breed.otterhound : 162,
  Breed.gordon_setter : 161,
  Breed.miniature_schnauzer : 161,
  Breed.shetland_sheepdog : 161,
  Breed.cocker_spaniel : 160,
  Breed.komondor : 160,
  Breed.miniature_poodle : 160,
  Breed.staffordshire_bullterrier : 160,
  Breed.walker_hound : 155,

  # Cats
  Breed.russian_blue : 1246,
  Breed.bombay : 1235,
  Breed.snowshoe : 1125,
  Breed.american_bobtail : 1075,
  Breed.maine_coon : 918,
  Breed.turkish_van : 815,
  Breed.himalayan : 801,
  Breed.turkish_angora : 750,
  Breed.norwegian_forest_cat : 580,
  Breed.british_shorthair : 567,
  Breed.american_shorthair : 530,
  Breed.oriental_short_hair : 491,
  Breed.exotic_shorthair : 471,
  Breed.persian : 402,
  Breed.tortoiseshell : 397,
  Breed.scottish_fold : 380,
  Breed.calico : 347,
  Breed.burmese : 344,
  Breed.torbie : 340,
  Breed.dilute_calico : 323,
  Breed.tuxedo : 319,
  Breed.dilute_tortoiseshell : 316,
  Breed.egyptian_mau : 305,
  Breed.tabby : 302,
  Breed.siamese : 289,
  Breed.ragdoll : 267,
  Breed.tonkinese : 260,
  Breed.abyssinian : 255,
  Breed.balinese : 254,
  Breed.bengal : 248,
  Breed.tiger : 226,
  Breed.manx : 206
}

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

def conform_img(img_data):
  if img_data.shape[2] == 4:
    return img_as_ubyte(rgba2rgb(img_data))
  return img_data

def preprocess_img_cat(img_path, bb: BoundingBox=None):
  img_data = io.imread(img_path)
  img_w = img_data.shape[1] // 6
  img_h = img_data.shape[0] // 6
  img_data = img_data[img_h:img_data.shape[0]-img_h, img_w:img_data.shape[1]-img_w]
  return conform_img(img_as_ubyte(resize(img_data, (64,64), anti_aliasing=True)))

def preprocess_img_dog(img_path, bb: BoundingBox):
  img_data = io.imread(img_path)
  img_data = img_data[bb.y:bb.y+bb.h, bb.x:bb.x+bb.w]
  return conform_img(img_as_ubyte(resize(img_data, (64,64), anti_aliasing=True)))

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
  breed_counts = {}

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # Cats
  n_cat = 0
  for sample in orig_data.cat_breeds:
    img_path = sample[0]
    l0 = Species.cat
    l1 = sample[1]
    if l1 not in BREED_COUNTS_TOTAL:
      continue
    if l1 not in breed_counts:
      breed_counts[l1] = 1
    else:
      if breed_counts[l1] >= BREED_COUNTS_TOTAL[l1]:
        continue
      else:
        breed_counts[l1] += 1
    if n_cat % 1000 == 0:
      print(f"Preprocessing cat {n_cat} {os.path.basename(img_path)}")
    proc_img = preprocess_img_cat(img_path)
    if proc_img.shape != (64,64,3):
      print(f"WARNING: {img_path} dimensions were unexpected: {proc_img.shape}")
    new_filename = str(uuid.uuid4())
    out_path = os.path.join(out_dir, new_filename + ".png")
    io.imsave(out_path, proc_img)
    samples.append({"img" : new_filename, "species" : l0.value, "breed" : l1.value})
    n_cat += 1

  # Dogs
  n_dog = 0
  for sample in orig_data.dog_breeds:
    img_path = sample[0]
    bound = sample[1]
    if bound.w < 1 or bound.h < 1:
      continue
    l0 = Species.dog
    l1 = sample[2]
    if l1 not in BREED_COUNTS_TOTAL:
      continue
    if l1 not in breed_counts:
      breed_counts[l1] = 1
    else:
      if breed_counts[l1] >= BREED_COUNTS_TOTAL[l1]:
        continue
      else:
        breed_counts[l1] += 1
    if n_dog % 1000 == 0:
      print(f"Preprocessing dog {n_dog} {os.path.basename(img_path)}")
    proc_img = preprocess_img_dog(img_path, bb=bound)
    if proc_img.shape != (64,64,3):
      print(f"WARNING: {img_path} dimensions were unexpected: {proc_img.shape}")
    new_filename = str(uuid.uuid4())
    out_path = os.path.join(out_dir, new_filename + ".png")
    io.imsave(out_path, proc_img)
    samples.append({"img" : new_filename, "species" : l0.value, "breed" : l1.value})
    n_dog += 1

  return pd.DataFrame(samples), n_cat, n_dog

def data_preprocess(in_dir, out_dir):
  print(f"Performing data_preprocess('{in_dir}','{out_dir}')")
  orig_data = read_orig(in_dir)
  print(f"Found {len(orig_data.cat_breeds)} cat samples")
  print(f"Found {len(orig_data.dog_breeds)} dog samples")
  new_data, n_cat, n_dog = preprocess_orig(orig_data, out_dir)
  dataset_out_path = os.path.join(out_dir, LABELS_FILENAME)
  new_data.to_pickle(dataset_out_path)
  print(f"Finished preprocessing {n_cat} cat samples and {n_dog} dog samples, labels written to {dataset_out_path}")

def data_a1(in_dir: Path, out_dir: Path):
  print(f"Performing data_a1('{in_dir}','{out_dir}')")
  dataset_path = os.path.join(in_dir, LABELS_FILENAME)
  dataset_out_path = os.path.join(out_dir, LABELS_FILENAME)
  data = pd.read_pickle(dataset_path)
  print(f"Processing {data.shape[0]} samples")

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for idx, row in data.iterrows():
    if idx % 1000 == 0:
      print(f"Working on sample {idx}")
    img = row["img"]
    species = row["species"]
    breed = row["breed"]
    img_path = os.path.join(in_dir, f"{img}.png")
    img_data = io.imread(img_path)
    a1_data = a1(img_data)
    out_path = os.path.join(out_dir, f"{img}.npy")
    np.save(out_path, a1_data)

  data.to_pickle(dataset_out_path)
  print(f"Finished processing samples. Results written to {out_dir}")

def data_a2(in_dir: Path, out_dir: Path):
  print(f"Performing data_a2('{in_dir}','{out_dir}')")
  dataset_path = os.path.join(in_dir, LABELS_FILENAME)
  dataset_out_path = os.path.join(out_dir, LABELS_FILENAME)
  data = pd.read_pickle(dataset_path)
  print(f"Processing {data.shape[0]} samples")

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for idx, row in data.iterrows():
    if idx % 1000 == 0:
      print(f"Working on sample {idx}")
    img = row["img"]
    species = row["species"]
    breed = row["breed"]
    img_path = os.path.join(in_dir, f"{img}.png")
    img_data = io.imread(img_path)
    a2_data = a2(img_data)
    out_path = os.path.join(out_dir, f"{img}.npy")
    np.save(out_path, a2_data)

  data.to_pickle(dataset_out_path)
  print(f"Finished processing samples. Results written to {out_dir}")

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