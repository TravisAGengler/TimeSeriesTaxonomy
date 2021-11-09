import pandas as pd

from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

Path = str
Label = str
Dataset = pd.DataFrame

class Species(Enum):
  dog = 1
  cat = 2

class Breed(Enum):
  # Dog breeds
  affenpinscher = 1
  afghan_hound = 2
  african_hunting_dog = 3
  airedale = 4
  american_staffordshire_terrier = 5
  appenzeller = 6
  australian_terrier = 7
  basenji = 8
  basset = 9
  beagle = 10
  bedlington_terrier = 11
  bernese_mountain_dog = 12
  black_and_tan_coonhound = 13 # black-and-tan_coonhound
  blenheim_spaniel = 14
  bloodhound = 15
  bluetick = 16
  border_collie = 17
  border_terrier = 18
  borzoi = 19
  boston_bull = 20
  bouvier_des_flandres = 21
  boxer = 22
  brabancon_griffon = 23
  briard = 24
  brittany_spaniel = 25
  bull_mastiff = 26
  cairn = 27
  cardigan = 28
  chesapeake_bay_retriever = 29
  chihuahua = 30
  chow = 31
  clumber = 32
  cocker_spaniel = 33
  collie = 34
  curly_coated_retriever = 35 # curly-coated_retriever
  dandie_dinmont = 36
  dhole = 37
  dingo = 38
  doberman = 39
  english_foxhound = 40
  english_setter = 41
  english_springer = 42
  entlebucher = 43
  eskimo_dog = 44
  flat_coated_retriever = 45 # flat-coated_retriever
  french_bulldog = 46
  german_shepherd = 47
  german_short_haired_pointer = 48 # german_short-haired_pointer 
  giant_schnauzer = 49
  golden_retriever = 50
  gordon_setter = 51
  great_dane = 52
  great_pyrenees = 53
  greater_swiss_mountain_dog = 54
  groenendael = 55
  ibizan_hound = 56
  irish_setter = 57
  irish_terrier = 58
  irish_water_spaniel = 59
  irish_wolfhound = 60
  italian_greyhound = 61
  japanese_spaniel = 62
  keeshond = 63
  kelpie = 64
  kerry_blue_terrier = 65
  komondor = 66
  kuvasz = 67
  labrador_retriever = 68
  lakeland_terrier = 69
  leonberg = 70
  lhasa = 71
  malamute = 72
  malinois = 73
  maltese_dog = 74
  mexican_hairless = 75
  miniature_pinscher = 76
  miniature_poodle = 77
  miniature_schnauzer = 78
  newfoundland = 79
  norfolk_terrier = 80
  norwegian_elkhound = 81
  norwich_terrier = 82
  old_english_sheepdog = 83
  otterhound = 84
  papillon = 85
  pekinese = 86
  pembroke = 87
  pomeranian = 88
  pug = 89
  redbone = 90
  rhodesian_ridgeback = 91
  rottweiler = 92
  saint_bernard = 93
  saluki = 94
  samoyed = 95
  schipperke = 96
  scotch_terrier = 97
  scottish_deerhound = 98
  sealyham_terrier = 99
  shetland_sheepdog = 100
  shih_tzu = 101 #shih-tzu
  siberian_husky = 102
  silky_terrier = 103
  soft_coated_wheaten_terrier = 104 # soft-coated_wheaten_terrier
  staffordshire_bullterrier = 105
  standard_poodle = 106
  standard_schnauzer = 107
  sussex_spaniel = 108
  tibetan_mastiff = 109
  tibetan_terrier = 110
  toy_poodle = 111
  toy_terrier = 112
  vizsla = 113
  walker_hound = 114
  weimaraner = 115
  welsh_springer_spaniel = 116
  west_highland_white_terrier = 117
  whippet = 118
  wire_haired_fox_terrier = 119 # wire-haired_fox_terrier
  yorkshire_terrier = 120

  # Cat breeds
  abyssinian = 121 # Abyssinian
  canadian_hairless = 122 # Canadian Hairless
  extra_toes_cat_hemingway_polydactyl = 123 # Extra-Toes Cat - Hemingway Polydactyl
  oriental_short_hair = 124 # Oriental Short Hair
  somali = 125 # Somali
  american_bobtail = 126 # American Bobtail
  chartreux = 126 # Chartreux
  havana = 127 # Havana 
  oriental_tabby = 128 # Oriental Tabby
  sphynx_hairless_cat = 129 # Sphynx - Hairless Cat
  american_curl = 130 # American Curl
  chausie = 131 # Chausie
  himalayan = 132 # Himalayan
  persian = 133 # Persian
  tabby = 134 # Tabby
  american_shorthair = 135 # American Shorthair
  chinchilla = 136 # Chinchilla
  japanese_bobtail = 137 # Japanese Bobtail
  pixiebob = 138 # Pixiebob
  tiger = 139 # Tiger
  american_wirehair = 140 # American Wirehair
  cornish_rex = 141 # Cornish Rex
  javanese = 142 # Javanese
  ragamuffin = 143 # Ragamuffin
  tonkinese = 144 # Tonkinese
  applehead_siamese = 145 # Applehead Siamese
  cymric = 146 # Cymric
  korat = 147 # Korat
  ragdoll = 148 # Ragdoll
  torbie = 149 # Torbie
  balinese = 150 # Balinese
  devon_rex = 151 # Devon Rex
  la_perm = 152 # LaPerm
  russian_blue = 153 # Russian Blue
  tortoiseshell = 154 # Tortoiseshell
  bengal = 155 # Bengal
  dilute_calico = 156 # Dilute Calico
  maine_coon = 157 # Maine Coon
  scottish_fold = 158 # Scottish Fold
  turkish_angora = 159 # Turkish Angora
  birman = 160 # Birman
  dilute_tortoiseshell = 161 # Dilute Tortoiseshell
  manx = 162 # Manx
  selkirk_rex = 163 # Selkirk Rex
  turkish_van = 164 # Turkish Van
  bombay = 165 # Bombay
  domestic_long_hair = 166 # Domestic Long Hair
  munchkin = 167 # Munchkin
  siamese = 168 # Siamese
  tuxedo = 169 # Tuxedo
  british_shorthair = 170 # British Shorthair
  domestic_medium_hair = 171 # Domestic Medium Hair
  nebelung = 172 # Nebelung
  siberian = 173 # Siberian
  york_chocolate = 174 # York Chocolate
  burmese = 175 # Burmese
  domestic_short_hair = 176 # Domestic Short Hair
  norwegian_forest_cat = 177 # Norwegian Forest Cat
  silver = 178 # Silver
  burmilla = 179 # Burmilla
  egyptian_mau = 180 # Egyptian Mau
  ocicat = 181 # Ocicat
  singapura = 182 # Singapura
  calico = 183 # Calico
  exotic_shorthair = 184 # Exotic Shorthair
  oriental_long_hair = 185 # Oriental Long Hair
  snowshoe = 186 # Snowshoe

@dataclass
class BoundingBox:
  x: int
  y: int
  w: int
  h: int

@dataclass
class OrigDataset:
  cat_breeds: List[Tuple[Path, Label]]
  dog_breeds: List[Tuple[Path, BoundingBox, Label]]
