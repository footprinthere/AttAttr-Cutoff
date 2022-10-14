import sys
import os
import random
from collections import defaultdict


RATIO = (0.45, 0.05, 0.5)
dirname = sys.argv[1]


def get_label(line):
    return line.strip().split('\t')[3]


def get_index(line):
    return int(line.strip().split('\t')[0])


# Read dataset files
train = open(os.path.join(dirname, "train.tsv"), 'r')
dev = open(os.path.join(dirname, "dev.tsv"), 'r')

lines = train.readlines()
for line in dev.readlines()[1:]:
    parsed = line.split('\t')
    idx = int(parsed[0]) + 2490
    parsed[0] = str(idx)
    lines.append('\t'.join(parsed))

# Classify lines by labels
classified = defaultdict(list)
for line in lines[1:]:
    classified[get_label(line)].append(line)

# Split datasets
tr_lines = []
dev_lines = []
ts_lines = []

for set in classified.values():
    random.shuffle(set)
    tr_end = int(len(set) * RATIO[0])
    dev_end = int(len(set) * (RATIO[0]+RATIO[1]))
    tr_lines.extend(set[:tr_end])
    dev_lines.extend(set[tr_end:dev_end])
    ts_lines.extend(set[dev_end:])

tr_lines.sort(key=get_index)
dev_lines.sort(key=get_index)
ts_lines.sort(key=get_index)
tr_lines = [lines[0]] + tr_lines
dev_lines = [lines[0]] + dev_lines
ts_lines = [lines[0]] + ts_lines

new_train = open(os.path.join(dirname, "new_train.tsv"), 'w')
new_dev = open(os.path.join(dirname, "new_dev.tsv"), 'w')
new_test = open(os.path.join(dirname, "new_test.tsv"), 'w')

new_train.writelines(tr_lines)
new_dev.writelines(dev_lines)
new_test.writelines(ts_lines)

train.close()
dev.close()
new_train.close()
new_dev.close()
new_test.close()
