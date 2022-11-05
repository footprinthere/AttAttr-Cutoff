import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import random
import logging
import json

import numpy as np
import torch

from transformers_cutoff import (
    RobertaConfig,
    RobertaTokenizer,
)

from modeling_roberta import RobertaForSequenceClassification
from classifier_processer import MnliProcessor, InputFeatures

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelInput:
    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
    ):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.input_len = int(attention_mask[0].sum())


class AttrScoreGenerator:
    """
    Given an input example, generates self-attention attribution score
    with fine-tuned model.
    """

    num_labels_task_map = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "rte": 2,
        "sst-2": 2,
        "qqp": 2,
        "qnli": 2,
        "wnli": 2,
        "sts-b": 1,
    }

    def __init__(
        self,
        model_name,
        task_name,
        model_file=None,
        num_batches=4,
        batch_size=16,
        random_seed=1,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.model_file = model_file
        self.num_batches = num_batches
        self.batch_size = batch_size

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self.device = 'cuda'

        self.prepare_model()

    def genereate_attrscore(self, inputs):
        """
        input을 받아서 그에 대한 self-attention attribution score를 산출.
        Returns: `num_layers * [(num_heads, input_len, input_len)]`
        """

        num_layers, num_heads = 12, 12
        input_len = inputs.input_len

        model_outputs = model_outputs = self.model(
            input_ids=inputs.input_ids,
            token_type_ids=inputs.token_type_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.labels,
        )
        logits = model_outputs[1]
        attentions = model_outputs[2]       # num_layers * [batch_size, num_heads, seqlen, seqlen]
        pred_label = int(torch.argmax(torch.squeeze(logits, dim=0)))

        prob = torch.nn.functional.softmax(logits)

        att_all = []
        result_attr = []

        for layer in range(num_layers):
            att_all.append(attentions[layer])
            scaled_att, step = self.scale_input(attentions[layer])
            scaled_att.requires_grad_(True)

            attr_all = torch.zeros((self.batch_size, num_heads, input_len, input_len))

            # 원래 총 m(= num_batches * batch_size)번의 계산을 해야 하지만 batch 단위로 묶어서 수행
            for batch in range(self.num_batches):
                batch_att = scaled_att[batch*self.batch_size : (batch+1)*self.batch_size]
                # batch_att: 앞서 얻은 scale_att 중 특정 구간을 추출한 것
                #       수식에서 (k/m)A_h에 해당
                #       [batch_size, num_heads, seq_len, seq_len]
                
                # attention에 대한 모델 prediction의 gradient 계산
                gradients = self.model(
                    input_ids=inputs.input_ids,
                    token_type_ids=inputs.token_type_ids,
                    attention_mask=inputs.attention_mask,
                    labels=inputs.labels,
                    replace_attention=batch_att,
                    pred_label=pred_label,
                )
                # gradients: 모델이 예측한 label에 해당하는 softmax 값의, batch_att에 대한 gradient
                #       [batch_size, num_heads, seq_len, seq_len]
                #       TODO: 왜 gradient를 A_h가 아니라 (k/m)A에 대해 계산할까?

                gradients = torch.sum(gradients, dim=0)     # batch 차원을 따라 sum
                attr_all = torch.add(attr_all, gradients)
                # attr_all: gradient의 누적합 [num_heads, seq_len, seq_len]

            # gradient에 attention을 곱함
            attr_all = attr_all[:, :input_len, :input_len] * step[0, :input_len, :input_len]
            result_attr.append(attr_all.data)
            
        # result_attr: num_layer * [num_heads, input_len, input_len]
        return result_attr

    def prepare_model(self):
        config = RobertaConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels_task_map[self.task_name],
            output_attentions=True,
        )

        if self.model_file is not None:
            state_dict = torch.load(self.model_file)
        else:
            state_dict = None
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            state_dict=state_dict,
        )
        self.model.to(self.device)
        self.model.eval()

    # FIXME: for test
    def process_input(self, data_dir, example_index):
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.processor = MnliProcessor()

        label_list = self.processor.get_labels()

        torch.cuda.empty_cache()

        eval_segment = "dev_matched"
        eval_examples = [
            self.processor.get_dev_examples(data_dir, segment=eval_segment)[example_index]
        ]

        max_len = 128
        eval_features, tokenlist = convert_examples_to_features(
            eval_examples, label_list, max_len, self.tokenizer
        )
        eval_feature = eval_features[0]

        input_ids = torch.tensor([eval_feature.input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([eval_feature.segment_ids], dtype=torch.long).to(self.device)
        label_ids = torch.tensor([eval_feature.label_id], dtype=torch.long).to(self.device)

        return ModelInput(input_ids, segment_ids, input_mask, label_ids)

    # FIXME: for test
    @classmethod
    def dump_attr(cls, attr, output_dir):
        file_name = "attr.json"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, file_name), 'w') as file:
            for layer_attr in attr:
                output = json.dumps(layer_attr.tolist())
                file.write(output + '\n')

        for i in range(len(attr)):
            print(i)
            print(attr[i].size())
            print()

    def scale_input(self, attention):
        """
        A (attention)을 받아서 (k/m)A를 쉽게 사용할 수 있도록 미리 계산해둠.
        """

        # baseline 설정 (default: zero tensor)
        baseline = torch.zeros_like(attention)
        # attention, baseline: [num_heads, seq_len, seq_len]

        # 전체 batch에 포함되어 있는 example의 총 개수 계산 후 그 수치로 정규화
        num_points = self.batch_size * self.num_batches
        scale = 1.0 / num_points

        # attribution matrix와 설정된 baseline의 차이를 계산
        step = (attention.unsqueeze(0) - baseline.unsqueeze(0)) * scale
        # 그 차이에 1/m부터 시작해서 1까지를 곱한 결과를 cat으로 연결
        result = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
        # result: [batch_size*num_batches, num_heads, seq_len, seq_len]

        return result, step[0]

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    Loads a data file into a list of `InputBatch`s.
    BERT의 입력 형식에 맞게 데이터를 적절히 변형하고 tokenize 수행.
    """

    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None

    features = []
    tokenslist = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        base_tokens = ["[UNK]"] + ["[UNK]"]*len(tokens_a) + ["[UNK]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            base_tokens += ["[UNK]"]*len(tokens_b) + ["[UNK]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        baseline_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(baseline_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if label_map:
            label_id = label_map[example.label]
        else:
            label_id = float(example.label)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          baseline_ids=baseline_ids))
        tokenslist.append({"token":tokens, "golden_label":example.label, "pred_label":None})
    return features, tokenslist


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.
    두 개의 문장이 주어지는 task에 대해, 문장쌍을 적절히 잘라 max_length를 준수하도록 함.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
