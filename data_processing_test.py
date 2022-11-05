"""
Cutoff repo 내부에 따로 구현되어 있는 데이터 처리 과정과 tokenizer의 동작을
그대로 재현할 수 있는지 테스트하는 프로그램
"""

from transformers_cutoff import RobertaTokenizer, GlueDataset, GlueDataTrainingArguments
from transformers_cutoff.data.data_collator import DefaultDataCollator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import logging


logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()


tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print("vocab size:", len(tokenizer.get_vocab()))

data_args = GlueDataTrainingArguments()
data_args.task_name = 'mnli'
data_args.data_dir = '/home/jovyan/work/datasets/MNLI'
train_dataset = GlueDataset(data_args, tokenizer)

train_sampler = RandomSampler(train_dataset)
collator = DefaultDataCollator()
data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    sampler=train_sampler,
    collate_fn=collator.collate_batch,
)
itr = iter(data_loader)

inputs = itr.next()
print(inputs.keys())

input_ids = inputs['input_ids']
print("input_ids:", input_ids)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("tokens:", tokens)
