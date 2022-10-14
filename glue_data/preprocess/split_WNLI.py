import sys
import os
import random
from collections import defaultdict


dirname = sys.argv[1]
dev_size = int(sys.argv[2])


def get_label(line):
    return line.strip().split('\t')[3]


def get_index(line):
    return int(line.strip().split('\t')[0])


train = open(os.path.join(dirname, "train.tsv"), 'r')
new_train = open(os.path.join(dirname, "new_train.tsv"), 'w')
new_dev = open(os.path.join(dirname, "new_dev.tsv"), 'w')

lines = train.readlines()
classified = defaultdict(list)

for line in lines[1:]:
    classified[get_label(line)].append(line)

tr_lines = []
dev_lines = []

for label in classified.keys():
    random.shuffle(classified[label])
    tr_lines.extend(classified[label][dev_size:])
    dev_lines.extend(classified[label][:dev_size])

tr_lines.sort(key=get_index)
dev_lines.sort(key=get_index)
tr_lines = [lines[0]] + tr_lines
dev_lines = [lines[0]] + dev_lines

new_train.writelines(tr_lines)
new_dev.writelines(dev_lines)

train.close()
new_train.close()
new_dev.close()
