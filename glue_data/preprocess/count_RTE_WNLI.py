import sys
import os
from collections import defaultdict


dirname = sys.argv[1]
filelist = os.listdir(dirname)

for filename in filelist:
    filepath = os.path.join(dirname, filename)
    file = open(filepath, 'r')
    lines = file.readlines()
    label_count = defaultdict(int)

    for line in lines[1:]:
        idx, s1, s2, label = line.strip().split('\t')
        label_count[label] += 1

    print(f"[{filepath}]\t total:", sum(label_count.values()), end='\t')
    print(dict(label_count))

    file.close()
