from transformers_cutoff import RobertaTokenizer, GlueDataset, GlueDataTrainingArguments
from transformers_cutoff.data.data_collator import DefaultDataCollator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import os

from transformers.data.processors.utils import InputFeatures


def main():

    # Cutoff의 trainer.train() 내부에서 진행되는 데이터 준비 과정 재현
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
    item = itr.next()

    print(item)


if __name__ == '__main__':
    main()
