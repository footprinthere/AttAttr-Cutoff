# MRPC, STS-B, QQP

import random
import pandas as pd
import seaborn as sns
from collections import defaultdict
    
def check_data_label(data_path, label_index):
    label_dict = defaultdict(int)
    idx_dict = {
        '1' : [],
        '0' : []
    }
    
    f = open(data_path, 'r')
    lines = f.readlines()
    
    for idx, line in enumerate(lines[1:]):
        values = line.strip().split('\t')
        label = values[label_index]
        
        if label not in ['0', '1']:
            label = label.split('?')[-1].strip() # wrong separator

        label_dict[label] += 1
        idx_dict[label].append(idx)
            
    print(f" size : {idx + 1}")
    
    for k, v in dict(label_dict).items():
        print(f" {k} : {v}")
    
    f.close()
    
    return idx_dict


def check_score_distribution(data_path, label_index):
    score_dict = defaultdict(int)
    
    f = open(data_path, 'r')
    lines = f.readlines()
    
    for idx, line in enumerate(lines[1:]):
        values = line.strip().split('\t')
        label = values[label_index]
        
        if label not in ['0', '1']:
            score = label.split('?')[-1].strip() # wrong separator

        score_dict[label] += 1
    
    sorted_scores = sorted(score_dict.items(), key = lambda item : item[0])
    score_df = pd.DataFrame.from_records(sorted_scores, columns = ["score", "cnt"])
    dist_plot = sns.barplot(data=score_df, x="score", y="cnt")
    fig = dist_plot.get_figure()
    # fig.savefig(f'{data_path} distribution.png')
            
    print(f" size : {idx + 1}")
    
    f.close()

def split_data(data_path, index_list, new_train_data_path, new_dev_data_path):
    f = open(data_path, 'r')
    lines = f.readlines()[1:]
    
    new_train_list = []
    new_dev_list = []
    
    for idx, line in enumerate(lines):
        if idx in index_list:
            new_dev_list.append(line)
        else:
            new_train_list.append(line)
        
    new_train = open(new_train_data_path, 'w')
    new_dev = open(new_dev_data_path, 'w')
    
    new_train.writelines(new_train_list)
    new_dev.writelines(new_dev_list)
    
    f.close()
    new_train.close()
    new_dev.close()



if __name__ == '__main__':
    file_dict = {
        'MRPC' : ['msr_paraphrase_test.txt', 'msr_paraphrase_train.txt'],
        'STS-B' : ['train.tsv', 'dev.tsv'],
        'QQP' : ['train.tsv', 'dev.tsv']

    }
    
    new_file_dict = {
        'MRPC' : ['new_train.txt', 'new_dev.txt', 'new_test.txt'],
        'STS-B' : ['new_train.tsv', 'new_dev.tsv', 'new_test.tsv'],
        'QQP' : ['new_train.tsv', 'new_dev.tsv', 'new_test.tsv']
    }
    
    # #MRPC
    print("\n-----MRPC-----")
    
    for fn in file_dict['MRPC']:
        print(fn)
        file_path = f'../MRPC/{fn}'
        
        if 'train' in fn:
            mrpc_idx_list = check_data_label(file_path, 0)
        else:
            check_data_label(file_path, 0)
        print()
    
    # QQP
    print("\n-----QQP-----")
    
    for fn in file_dict['QQP']:
        print(fn)
        file_path = f'../QQP/{fn}'
        if 'train' in fn:
            qqp_idx_list = check_data_label(file_path, -1)
        else:
            check_data_label(file_path, -1)
        print()
    
    # STS-B
    print("\n-----STS-B-----")
    
    for fn in file_dict['STS-B']:
        print(fn)
        file_path = f'../STS-B/{fn}'
        check_score_distribution(file_path, -1)
        print()
    
    # # MRPC split
    # mrpc_pos_idx = random.sample(mrpc_idx_list['1'], k=1000)
    # mrpc_neg_idx  = random.sample(mrpc_idx_list['0'], k=500)
    
    # split_data(f'../MRPC/msr_paraphrase_train.txt',
    #            mrpc_pos_idx + mrpc_neg_idx, 
    #            f'../MRPC/new_train.txt', 
    #            f'../MRPC/new_dev.txt'
    #            )

    # #QQP split
    # qqp_pos_idx = random.sample(qqp_idx_list['1'], k=25000)
    # qqp_neg_idx  = random.sample(qqp_idx_list['0'], k=15000)
    # split_data(f'../QQP/train.tsv',
    #            qqp_pos_idx + qqp_neg_idx, 
    #            f'../QQP/new_train.tsv', 
    #            f'../QQP/new_dev.tsv'
    #            )
    
    # # STS-B
    # sts_dev_idx = random.sample(list(range(0, 5750)), k=1500)
    # split_data(f'../STS-B/train.tsv',
    #            sts_dev_idx,
    #            f'../STS-B/new_train.tsv',
    #            f'../STS-B/new_dev.tsv'
    #            )
   
    