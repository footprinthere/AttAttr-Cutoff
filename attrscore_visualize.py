import sys
import os
import argparse
import torch
import numpy as np
from typing import List
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..")
sys.path.append(".")

from captum.attr import visualization

from transformers_cutoff import RobertaTokenizer, GlueDataset, GlueDataTrainingArguments
from transformers_cutoff import RobertaConfig, AutoTokenizer
# from modeling_roberta import RobertaForSequenceClassification
from transformers_cutoff import TrainingArguments
from transformers_cutoff import PreTrainedModel
from transformers_cutoff import glue_convert_examples_to_features, glue_output_modes, glue_processors
from transformers_cutoff import DefaultDataCollator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from attattr import AttrScoreGenerator, ModelInput
print(__name__)


def main():
    print("main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    

    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--attr_layer_strategy', type=str, default='max')
    parser.add_argument('--attr_mean_of_last_layers', type=int, default=2)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='visualize_results/')
    args = parser.parse_args()

    data_args = GlueDataTrainingArguments()
    args.task_name = args.task_name.lower()
    data_args.task_name = args.task_name
    data_args.data_dir = args.data_dir
    train_dataset = GlueDataset(data_args, tokenizer)

    train_sampler = RandomSampler(train_dataset)
    collator = DefaultDataCollator()
    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=collator.collate_batch,
    )
    
    task_dir = args.task_name.upper()
    if task_dir == "COLA":
        task_dir = "CoLA"
    generator = AttrScoreGenerator(
        model_name=args.model_name_or_path,
        task_name=args.task_name,
        model_file=f'/home/jovyan/work/checkpoint/{task_dir}/checkpoint_token/pytorch_model.bin',
    )
    print("itr")
    itr = iter(data_loader)
    while True:
        try:
            inputs = next(itr)
            #print(inputs)
            generate_visualization(inputs, generator, args, device,tokenizer)
        except StopIteration:
            break
    inputs = itr.next()

    
    
def generate_visualization(
    inputs,
    generator: AttrScoreGenerator,
    args,
    device,
    tokenizer,
):
    input_ids=inputs['input_ids'].to(device)
    model_input = ModelInput(
        input_ids=input_ids,
        token_type_ids=None,            # None for test; 실제로는 segment id 입력
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
    cls_attr = (cls_attr.max() - cls_attr) / (cls_attr.max() - cls_attr.min())
    example_token_ids = input_ids.squeeze(0)[:5]
    #[j for input_ids(i) != 1]
    print(example_token_ids,"\n")
    print(cls_attr,"\n")
    
    #example_token_id = [0] + (input_ids[i]*example_mask)[(input_ids[i]*example_mask).nonzero()]
    tokens = [e for e in tokenizer.convert_ids_to_tokens(example_token_ids)]
    tokens = [word.strip('Ġ') for word in tokens] 
    print(tokens)
    vis_data_records = [visualization.VisualizationDataRecord(cls_attr,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        tokens,
                                                        1)]
    html = visualization.visualize_text(vis_data_records)
    
    with open(f"{args.output_dir}/visualize.html", "a+") as h:
        h.write(html.data)
    

if __name__ == "__main__":
    main()