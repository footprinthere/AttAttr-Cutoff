import os
from .attrscore_generator import AttrScoreGenerator
from .model_input import ModelInput

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # using GPU 0


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
    inputs = itr.next()

    # 모델 입력을 ModelInput 객체로 결합
    model_input = ModelInput(
        input_ids=inputs['input_ids'],
        token_type_ids=None,            # None for test; 실제로는 segment id 입력
        attention_mask=inputs['attention_mask'],
        labels=inputs['labels'],
    )
    # 입력되는 모든 tensor의 첫 번째 차원(batch_size)은 1이어야 함에 유의

    # Attribution score generator 선언
    generator = AttrScoreGenerator(
        model_name='roberta-base',
        task_name='mnli',
        model_file='/home/jovyan/work/checkpoint/MNLI/checkpoint_token/pytorch_model.bin',
    )

    # Attribution score 생성
    attribution = generator.generate_attrscore(model_input)

    for i in range(len(attribution)):
        print(i)
        print(attribution[i].size())

    # generator.dump_attr() 사용해 json 파일로 저장 가능


if __name__ == '__main__':
    main()
