import os
import logging
import argparse

import torch

import numpy as np
from tqdm import tqdm

from transformers_cutoff import RobertaTokenizer, GlueDataset, GlueDataTrainingArguments, InputFeatures

from attattr import AttrScoreGenerator, ModelInput


logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()

CUTOFF_RATIO = [0.1, 0.05]
CUTOFF_RATIO_SUFFIX = ["10", "05"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--saved_npy_10', type=str, default=None)
    parser.add_argument('--saved_npy_05', type=str, default=None)
    parser.add_argument('--skip_first_n_examples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--save_period', type=int, default=500)
    parser.add_argument('--attr_layer_strategy', choices=['max', 'normalize'], default='norm')
    parser.add_argument('--model_max_length', type=int, default=128)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.save_dir, "temp")):
        os.makedirs(os.path.join(args.save_dir, "temp"), exist_ok=True)

    # Prepare DataLoader
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    data_args = GlueDataTrainingArguments()
    data_args.task_name = args.task_name.lower()
    data_args.data_dir = f'/home/jovyan/work/datasets/{args.task_name}'
    train_dataset = GlueDataset(data_args, tokenizer)

    logger.info("Data prepared")

    # Construct Generator
    generator = AttrScoreGenerator(
        model_name="roberta-base",
        task_name=args.task_name.lower(),
        model_file=f'/home/jovyan/work/checkpoint/{args.task_name}/checkpoint_token/pytorch_model.bin'
    )

    # Initialize np array
    np_arrays = []

    if args.saved_npy_10 and args.saved_npy_05:
        # Load from saved files
        np_arrays.append(np.load(args.saved_npy_10))
        np_arrays.append(np.load(args.saved_npy_05))
        logger.info(f"Resumed from saved npy files; Shape: {np_arrays[0].shape}, {np_arrays[1].shape}")
    else:
        for ratio in CUTOFF_RATIO:
            max_cutoff_length = int(args.model_max_length * ratio)
            array = np.zeros((len(train_dataset), max_cutoff_length))
            array.fill(-1)
            np_arrays.append(array)
        logger.info(f"Initialized new numpy arrays; Shape: {np_arrays[0].shape}, {np_arrays[1].shape}")


    for i, data in tqdm(enumerate(train_dataset), desc='Examples', total=len(train_dataset)):
        if args.skip_first_n_examples:
            if i < args.skip_first_n_examples:
                continue
            elif i == args.skip_first_n_examples:
                logger.info(f"Skipped first {i} examples")
                logger.info("Last rows:")
                for array in np_arrays:
                    logger.info(array[i-1])
        
        cutoff_indices = get_cutoff_indices(generator, data, args)
        for array, indices in zip(np_arrays, cutoff_indices):
            array[data.example_index, :len(indices)] = indices

        # Save temporary results
        if (i+1) % args.save_period == 0:
            logger.info(f"Saving numpy arrays after {i+1} examples")
            for r in range(len(cutoff_indices)):
                np_save(
                    dir=os.path.join(args.save_dir, "temp"),
                    file_name=f"{args.task_name}_indices_{i+1}_{CUTOFF_RATIO_SUFFIX[r]}",
                    arr=np_arrays[r],
                )

    # Save final results
    for r in range(len(cutoff_indices)):
        np_save(
            dir=args.save_dir,
            file_name=f"{args.task_name}_indices_{CUTOFF_RATIO_SUFFIX[r]}",
            arr=np_arrays[r],
        )
    logger.info("Successfully saved cache arrays")


def get_cutoff_indices(generator: AttrScoreGenerator, data: InputFeatures, args):
    model_input = ModelInput(
        input_ids=torch.tensor(data.input_ids).long().unsqueeze(0).to(generator.device),
        token_type_ids=None,
        attention_mask=torch.tensor(data.attention_mask).long().unsqueeze(0).to(generator.device),
        labels=torch.tensor(data.label).unsqueeze(0).to(generator.device),
    )

    attr = generator.generate_attrscore(model_input)

    # Compress attribution scores according to the selected strategy
    attr = torch.stack(attr).mean(dim=1)            # mean along head dimension

    if args.attr_layer_strategy == "max":
        attr = attr.max(dim=0).values               # max along layer dimension
    elif args.attr_layer_strategy == "normalize":
        attr = torch.sub(attr, attr.mean(dim=0))
        attr = torch.div(attr, attr.var(dim=0))
        attr = attr.max(dim=0).values               # max along layer dimension

    cls_attr = attr[0]

    # Sort tokens by attribution
    sorted_indices = torch.topk(cls_attr, k=int(max(CUTOFF_RATIO) * model_input.input_len), largest=False).indices
    sorted_indices = sorted_indices.cpu().numpy()

    cutoff_indices = []
    for ratio in CUTOFF_RATIO:
        cutoff_length = int(ratio * model_input.input_len)
        cutoff_indices.append(sorted_indices[:cutoff_length])
    
    return cutoff_indices


def np_save(dir, file_name, arr):
    np.save(os.path.join(dir, file_name), arr)


if __name__ == '__main__':
    main()
