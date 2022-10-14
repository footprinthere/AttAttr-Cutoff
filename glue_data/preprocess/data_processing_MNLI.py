

#! 1. 장르, 라벨을 같은 비율로 나눠야 함

import random

def create_val_set(type):
  """
  type: "matched" | "mismatched"
  """
  f = open(f"dev_{type}.tsv", "r")
  data = f.read().split("\n")
  header = data[0]
  genre_col = 3
  label_col = -1 # ref: transformers/data/processors/glue.py - MnliProcessor._create_examples
  labels = ["contradiction", "entailment", "neutral"]
  label_cnt = {"contradiction": 0, "entailment": 0, "neutral": 0}
  genres = []
  for i, line in enumerate(data[1:]):
    items = line.split("\t")
    if items[label_col] not in labels:
      print(f"Invalid label: [{i}] {items[label_col]}")
      continue
    label_cnt[items[label_col]] = label_cnt[items[label_col]]+1
    if items[genre_col] not in genres:
      genres.append(items[genre_col])

  print(label_cnt)
  print("Genres:")
  print(genres)

  # initialize dictionaries
  label_genre_cnt = {}
  label_genre_idx = {}
  for label in labels:
    label_genre_cnt[label] = {}
    label_genre_idx[label] = {}
    for genre in genres:
      label_genre_cnt[label][genre] = 0
      label_genre_idx[label][genre] = []

  for i, line in enumerate(data[1:]):
    items = line.split("\t")
    if items[label_col] not in labels:
      continue
    label_genre_cnt[items[label_col]][items[genre_col]] = label_genre_cnt[items[label_col]][items[genre_col]]+1
    label_genre_idx[items[label_col]][items[genre_col]].append(i)

  print(label_genre_cnt)

  random.seed(100)
  label_genre_val_idx = {}
  for k, v in label_genre_idx.items():
    label_genre_val_idx[k] = {}
    for ki, vi in v.items():
      n = label_genre_cnt[k][ki]//2
      label_genre_val_idx[k][ki] = random.sample(vi, n)

  idx_val = []
  for k, v in label_genre_val_idx.items():
    for ki, vi in v.items():
      idx_val += vi

  new_val_string = header + "\n"
  new_dev_string = header + "\n"

  for i, line in enumerate(data[1:]):
    if i in idx_val:
      new_val_string += line
      new_val_string += "\n"
    elif line != "":
      new_dev_string += line
      new_dev_string += "\n"

  f_new_val = open(f"./new_val_{type}.tsv", "w")
  f_new_val.write(new_val_string)
  f_new_dev = open(f"./new_dev_{type}.tsv", "w")
  f_new_dev.write(new_dev_string)

  f.close()
  f_new_val.close()
  f_new_dev.close()

def main():
  print("Matched!!!")
  create_val_set("matched")
  print("Mismatched!!!")
  create_val_set("mismatched")

main()