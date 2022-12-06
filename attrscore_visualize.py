import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from typing import List
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..")
sys.path.append(".")

from captum.attr import visualization

from transformers_cutoff import RobertaTokenizer, GlueDataset, GlueDataTrainingArguments
from transformers_cutoff import DefaultDataCollator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from attattr import AttrScoreGenerator, ModelInput
from transexp_orig.ExplanationGenerator import Generator
import save_attr_npy


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = 'cuda'
    
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--score_strategy', type=str, default='attattr')
    parser.add_argument('--attr_layer_strategy', type=str, default='max')
    parser.add_argument('--attr_mean_of_last_layers', type=int, default=2)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--data_dir', type=Path, default=Path('/home/jovyan/work/datasets'))
    parser.add_argument('--pretrain_dir', type=Path, default=Path('/home/jovyan/work/checkpoint'))
    parser.add_argument('--output_dir', type=str, default='visualize_results/')
    args = parser.parse_args()

    data_args = GlueDataTrainingArguments()
    args.task_name = args.task_name.lower()
    data_args.task_name = args.task_name
    data_args.data_dir = f'/home/jovyan/work/datasets/{args.task_name}'
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
    
    task_dir = args.task_name.upper()
    if task_dir == "COLA":
        task_dir = "CoLA"
        
    if args.score_strategy == "attattr":
        generator = AttrScoreGenerator(
            model_name=args.model_name_or_path,
            task_name=args.task_name,
            model_file=f'/home/jovyan/work/checkpoint/{task_dir}/checkpoint_token/pytorch_model.bin',
        )
        while True:
            try:
                inputs = next(itr)
                cls_attr, example_token_ids = attattr_score(inputs, generator, args, device)
                generate_visualization(cls_attr, example_token_ids, args, tokenizer)
            except StopIteration:
                break
    elif args.score_strategy == "transexp":
        ### TODO: need score from transexp
        model, features = save_attr_npy.get_model_and_data(save_attr_npy.get_task_name(args.task_name), tokenizer, args)
        explanations = Generator(model)
        for i in tqdm(range(len(features))):
            input_ids = torch.tensor(features[i].input_ids, dtype=int).reshape(1,-1).cuda()
            attention_masks = torch.tensor(features[i].attention_mask, dtype=torch.float32).reshape(1,-1).cuda()
            input_len = int(torch.sum(attention_masks, dim=1))
            expl, example_token_ids = transexp_score(explanations, input_ids, input_len, attention_masks)
            generate_visualization(expl, example_token_ids, args, tokenizer)
    else: raise ValueError(f"'score_strategy' must be one of ['attattr', 'transexp']; Got {args.score_strategy}")
    
    
def attattr_score(
    inputs,
    generator: AttrScoreGenerator,
    args,
    device,
):
    input_ids=inputs['input_ids'].to(device)
    model_input = ModelInput(
        input_ids=input_ids,
        token_type_ids=inputs.get('token_type_ids', None),            # None for test; 실제로는 segment id 입력
        attention_mask=inputs['attention_mask'].to(device),
        labels=inputs['labels'].to(device),
    )
    
    attr = generator.generate_attrscore(model_input)

    # Compress attribution scores according to the selected strategy
    attr = torch.stack(attr).mean(dim=1)            # mean along head dimension
            
    if args.attr_layer_strategy == "max":
        attr = attr.max(dim=0).values               # max along layer dimension
    
    elif args.attr_layer_strategy == "mean":
        if args.attr_mean_of_last_layers is None:
            n_layers = attr.size(0)
        elif args.attr_mean_of_last_layers > attr.size(0):
            raise ValueError("The value of the argument 'attr_mean_of_last_layers' exceeds boundary")
        else:
            n_layers = args.attr_mean_of_last_layers
        attr = attr[-n_layers:]                     # Select last n layers
        attr = attr.mean(dim=0)                     # mean along layer dimension
    
    elif args.attr_layer_strategy == "normalize":
        attr = torch.sub(attr, attr.mean(dim=0))
        attr = torch.div(attr, attr.var(dim=0))
        attr = attr.max(dim=0).values               # max along layer dimension

    else:
        raise ValueError(f"'attr_layer_strategy' must be one of ['max', 'mean', 'normalize']; Got {args.attr_layer_strategy}")
        
    cls_attr = attr[0]          # extract column for [CLS]
    input_ids = input_ids.squeeze(0)
    example_token_ids = input_ids[:model_input.input_len]
    return cls_attr, example_token_ids

def transexp_score(
    explanations,
    input_ids,
    input_len,
    attention_mask,
):
    example_token_ids = input_ids[0][:input_len]
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    expl = expl[:input_len]
    expl[0] = expl.max()

    return expl, example_token_ids
    
def generate_visualization(
    cls_attr,
    example_token_ids,
    args,
    tokenizer,
):
    
    #print("original cls_attr", cls_attr)
    expl = (cls_attr.max() - cls_attr) / (cls_attr.max() - cls_attr.min()) * 2.5 # range: [0,2.5]
    expl = torch.exp(expl) - 1 # range: [0,e^2.5-1]
    expl = torch.exp(expl) # range: [1,e^(e^2.5-1)]
    expl = (expl - expl.min()) / (expl.max() - expl.min()) # range: [0,1]
    #print("normalized cls_attr", expl)
    
    tokens = [e for e in tokenizer.convert_ids_to_tokens(example_token_ids)]
    tokens = [word.strip('Ġ') for word in tokens] 
    tokens = [word.replace('<s>', '[CLS/BOS]') for word in tokens]
    tokens = [word.replace('</s>', '[SEP/EOS]') for word in tokens]
    
    vis_data_records = [visualization.VisualizationDataRecord(
                                                        expl,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        tokens,
                                                        1)]
    html = visualization.visualize_text(vis_data_records)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.score_strategy == "attattr": output_file = f"{args.output_dir}/{args.task_name}-{args.score_strategy}-{args.attr_layer_strategy}-visualize.html"
    else: output_file = f"{args.output_dir}/{args.task_name}-{args.score_strategy}-visualize.html"
    with open(output_file, "a") as h:
        h.write(html.data)
        h.write("cls_attr: ")
        h.write(str([(tokens[i], cls_attr[i].item()) for i in range(len(tokens))]))
        h.write("<br>")
    
if __name__ == "__main__":
    main()