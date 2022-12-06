import os
import re
import argparse
from pathlib import Path

import torch
import transformers
import numpy as np
from tqdm import tqdm

from transformers_cutoff.trainer import Trainer #calculate_token_exp_idx
from transformers_cutoff import AutoTokenizer, AutoConfig, glue_tasks_num_labels
from transformers_cutoff.data.processors.glue import glue_processors
from transformers_cutoff.data.processors.utils import InputFeatures

from transexp_orig.RobertaForSequenceClassification import RobertaForSequenceClassification
from transexp_orig.ExplanationGenerator import Generator

# cls, pad, bos, eos, sep, unk, period
# TOKENS_EXCLUDE = list(set([
#   tokenizer.cls_token_id,
#   tokenizer.pad_token_id,
#   tokenizer.sep_token_id,
#   tokenizer.unk_token_id,
#   tokenizer.bos_token_id,
#   tokenizer.eos_token_id,
#   tokenizer.convert_tokens_to_ids(".")]))
TOKENS_EXCLUDE = [0,1,2,3,4]
TASKS = ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"]
MAX_SEQ_LENGTH = 128

transformers.enable_full_determinism(1)

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=Path, 
                      default=Path('/home/jovyan/work/datasets'))
  parser.add_argument('--pretrain_dir', type=Path, 
                      default=Path('/home/jovyan/work/checkpoint'))
  parser.add_argument('--save_dir', type=Path, 
                      default=Path('./cutoff_idx_npy'))
  parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                      type=str, default='all')
  parser.add_argument('--cutoff_ratio', help='cutoff ratio',
                      type=list, default=[0.1, 0.05])
  parser.add_argument('--min_cutoff_token', help='minimum cutoff token',
                      type=int, default=1)
  parser.add_argument('--gpu', help='gpu number',
                      type=str, default="1")
  parser.add_argument('--exclude_special_tokens', help='Exclude or include special token',
                      type=bool, default=True)
  
  return parser.parse_args()


def get_task_name(raw_task):
  task = raw_task.upper()
  if task == "COLA":
    task = "CoLA"
    
  return task


def get_model_and_data(task, tokenizer, args):
  print("+"*20 + task + "+"*20)
  
  pretrain_path = str(args.pretrain_dir / task / "checkpoint_token/")
  print(f"Get pretrained model from {pretrain_path}")  
  model = RobertaForSequenceClassification.from_pretrained(
            pretrain_path,
          ).to("cuda")
  model.eval()
  
  processor = glue_processors[task.lower()]()
  data_dir = str(args.data_dir / task)
  examples = (processor.get_train_examples(data_dir))
  batch_encoding = tokenizer.batch_encode_plus(
      [(example.text_a, example.text_b) for example in examples], 
      max_length=MAX_SEQ_LENGTH, 
      pad_to_max_length=True, 
      #return_token_type_ids=True,
  )
  
  label_list = processor.get_labels()
  label_map = {label: i for i, label in enumerate(label_list)}
  
  num_labels = glue_tasks_num_labels[task.lower()]
  if task == "STS-B":
    labels = [float(example.label) for example in examples]
    print("label: continuous label")
  else:
    labels = [label_map[example.label] for example in examples]
    print(f"label: {label_list}")
  print()
  
  features = []
  for i in range(len(examples)):
    inputs = {k: batch_encoding[k][i] for k in batch_encoding}

    feature = InputFeatures(**inputs, example_index=i, label=labels[i])
    features.append(feature)
    
  return model, features


def calculate_token_exp_idx(explanations, input_ids, input_embed, attention_mask, input_len, args):
  expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
  # normalize scores
  expl_copy = expl.data.cpu()
  del expl
  torch.cuda.empty_cache()

  expl = (expl_copy - expl_copy.min()) / (expl_copy.max() - expl_copy.min())
  
  zero_mask = torch.ones(input_embed.shape[0]).to('cuda')

  # assign additional cutoff length to consider special tokens which can be excluded
  input_id = input_ids[0][:input_len]

  expl = expl[:input_len]
  if args.exclude_special_tokens:
      expl_exclude = []
      for i in range(input_id.shape[0]):
          if input_id[i] in TOKENS_EXCLUDE:
              expl_exclude.append(100)
          else:
              expl_exclude.append(expl[i])
              
      expl = torch.tensor(expl_exclude)
      
      
  cutoff_idx_list = []
  
  for cutoff_ratio in args.cutoff_ratio:
    cutoff_length = int(input_len * cutoff_ratio)

    if cutoff_length < args.min_cutoff_token:
        cutoff_length = args.min_cutoff_token

    cutoff_idx_list.append(torch.topk(expl, cutoff_length, largest=False).indices)
      
  return cutoff_idx_list
    

def initialize_cutoff_index_array(dataset_size, args):
  saved_cutoff_idx_list = []
  
  for cutoff_ratio in args.cutoff_ratio:
    max_cutoff_length = int(MAX_SEQ_LENGTH * cutoff_ratio)
    saved_cutoff_idx = np.zeros((dataset_size, max_cutoff_length), dtype=int)
    saved_cutoff_idx.fill(-1)
    saved_cutoff_idx_list.append(saved_cutoff_idx.copy())
  return saved_cutoff_idx_list


def main():
  args = parse()
  
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
  
  tasks = TASKS if args.tasks == "all" else args.tasks.replace(" ", "").split(",")
  tokenizer = AutoTokenizer.from_pretrained("roberta-base")
  
  os.makedirs(str(args.save_dir), exist_ok=True)
  
  for task in tasks:
    task = get_task_name(task)
    
    model, features = get_model_and_data(task, tokenizer, args)

    explanations = Generator(model)
    
    saved_cutoff_idx_list = initialize_cutoff_index_array(len(features), args)
    str_exclude_st = "exclude" if args.exclude_special_tokens else "include"
    
    # for i in range(5):
    for i in tqdm(range(len(features))):
      
      if i%1000==0 and i!=0:
        print(f"save tmp file at step {i}")
        for idx_cr in range(len(args.cutoff_ratio)):
          np.save(str(args.save_dir / f"{task}_{str_exclude_st}_{args.cutoff_ratio[idx_cr]}-step{i}.npy"), saved_cutoff_idx_list[idx_cr])
          tmp_file = str(args.save_dir / f"{task}_{str_exclude_st}_{args.cutoff_ratio[idx_cr]}-step{i-1000}.npy")
          if os.path.exists(tmp_file):
            os.remove(tmp_file)
      
      input_ids = torch.tensor(features[i].input_ids, dtype=int).reshape(1,-1).cuda()
      token_type_ids = None if features[i].token_type_ids is None \
                        else torch.tensor(features[i].token_type_ids, dtype=int).reshape(1,-1).cuda()
      
      attention_masks = torch.tensor(features[i].attention_mask, dtype=torch.float32).reshape(1,-1).cuda()
      input_embeds = model.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids)
      input_len = int(torch.sum(attention_masks, dim=1))
      
      lowest_indices_list = calculate_token_exp_idx(explanations, input_ids, input_embeds, attention_masks, input_len, args)
      
      for idx_cr in range(len(args.cutoff_ratio)):
        cutoff_idx = lowest_indices_list[idx_cr].cpu().numpy()
        saved_cutoff_idx_list[idx_cr][features[i].example_index, :len(cutoff_idx)] = cutoff_idx

    print("saving npy files")
    for idx_cr in range(len(args.cutoff_ratio)):
      np.save(str(args.save_dir / f"{task}_{str_exclude_st}_{args.cutoff_ratio[idx_cr]}.npy"), saved_cutoff_idx_list[idx_cr])
    print("file saved")
    print("Deleting tmp file")
    for cutoff_ratio in args.cutoff_ratio:
      tmp_file_re = f"{task}_{str_exclude_st}_0\.[0-9]+-step[0-9]+\.npy"
      for file_name in os.listdir(str(args.save_dir)):
        if re.match(tmp_file_re, file_name):
          os.remove(str(args.save_dir / file_name))
    print("Done!")

if __name__=="__main__":
  main()