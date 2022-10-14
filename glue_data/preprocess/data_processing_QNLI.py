
#* 1. check number of lines in train and dev
#* 2. check distribution of labels in train and dev
#* 3. split train into two file - train and test - two files should have same label distribution and n(test)==n(dev)

import random

path = {
  "train": "./train.tsv",
  "dev": "./dev.tsv",
  "test": "./test.tsv",
  "new_train": "./new_train.tsv",
  "new_val": "./new_val.tsv",
}
label_col = 3

def initial_label_cnt():
  return {"entailment": 0, "not_entailment": 0}

def check_files(filename):
  f = open(filename, "r")
  data = f.read()

  data = data.split("\n")
  print(f"Length of {filename}: {len(data)}")

  label_cnt = initial_label_cnt()
  for i, line in enumerate(data):
    if i==0:
      continue

    items = line.split("\t")

    if len(items) < label_col:
      print(f"index: {i}")
      print(line)
      continue
    if not items[label_col] in label_cnt.keys():
      print(items[label_col])
      continue

    label_cnt[items[label_col]] = label_cnt[items[label_col]] + 1

  print(label_cnt)
  f.close()

  return label_cnt


def create_val_set(n_val):
  random.seed(100)

  train_file = open(path["train"], "r")
  train_data = train_file.read().split("\n")

  label_idx = {"entailment": [], "not_entailment": []}

  for i, line in enumerate(train_data):
    if i == 0:
      continue

    items = line.split("\t")

    if len(items) < label_col:
      continue
    if not items[label_col] in label_idx.keys():
      print(items[label_col])
      continue

    label_idx[items[label_col]].append(i)

  idx_entail = random.sample(label_idx["entailment"], n_val//2)
  idx_notentail = random.sample(label_idx["not_entailment"], n_val//2)
  idx_val = idx_entail + idx_notentail

  new_train = "index\tquestion\tsentence\tlabel\n"
  new_val = "index\tquestion\tsentence\tlabel\n"
  for i, line in enumerate(train_data):
    if i == 0 or i==len(train_data)-1:
      continue

    if i in idx_val:
      new_val += line
      new_val += "\n"
    else:
      new_train += line
      new_train += "\n"

  new_train_file = open(path["new_train"], "w")
  new_train_file.write(new_train)
  print("new_train.tsv created!")

  new_val_file = open(path["new_val"], "w")
  new_val_file.write(new_val)
  print("new_val.tsv created!")

  train_file.close()
  new_train_file.close()
  new_val_file.close()

def main():
  print("Step 1. Check number and label cnt")
  print("===dev===")
  check_files(path["dev"])
  print("===train===")
  check_files(path["train"])

  print("Step 2. Split train set into new_train and new_val")
  n_val = 5464 # manually set
  create_val_set(n_val)

  print("Step 3. Check created files")
  print("===new_train===")
  check_files(path["new_train"])
  print("===new_val===")
  check_files(path["new_val"])
  

main()
