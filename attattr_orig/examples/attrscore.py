import argparse
import logging
import random
import json
import os

import numpy as np
import torch

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
)
from examples.classifier_processer import (
    InputFeatures,
    MrpcProcessor,
    MnliProcessor,
    RteProcessor,
    ColaProcessor,
    SstProcessor,
    QqpProcessor,
    QnliProcessor,
    WnliProcessor,
    StsProcessor,
)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "sst-2": SstProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "wnli": WnliProcessor,
    "sts-b": StsProcessor,
}

num_labels_task = {
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


def scale_input(attention, batch_size, num_batches):
    """
    A (attention)을 받아서 (k/m)A를 쉽게 사용할 수 있도록 미리 계산해둠.
    """
    # TODO: k/m을 곱하는 작업을 dynamic 하게 하지 않고 전부 저장해둔 다음 slice 해서 사용하는 이유는 뭘까?

    # baseline 설정 (default: zero tensor)
    baseline = torch.zeros_like(attention)
    # attention, baseline: [num_heads, seq_len, seq_len]

    # 전체 batch에 포함되어 있는 example의 총 개수 계산 후 그 수치로 정규화
    num_points = batch_size * num_batches
    scale = 1.0 / num_points

    # attribution matrix와 설정된 baseline의 차이를 계산
    step = (attention.unsqueeze(0) - baseline.unsqueeze(0)) * scale
    # 그 차이에 1/m부터 시작해서 1까지를 곱한 결과를 cat으로 연결
    result = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
    # result: [batch_size*num_batches, num_heads, seq_len, seq_len]

    return result, step[0]


def generate_attrscore(
    model,          # 모델이 attention을 return 하도록 config가 설정되어 있어야 함
    tokenizer,
    processor,
    data_dir,
    example_index,
    batch_size=16,  # TODO: batch_size와 num_batch의 역할은 뭘까?
    num_batches=4,
    max_len=128,
    return_attentions=False,
    device="cuda",
):

    label_list = processor.get_labels()

    torch.cuda.empty_cache()

    eval_segment = "dev_matched" if task_name == "mnli" else "dev"
    eval_examples = [
        processor.get_dev_examples(data_dir, segment=eval_segment)[example_index]
    ]
    # FIXME: 여러 example 한 번에 처리하려면 수정 필요 (index로 slicing)

    model.eval()

    # FIXME: RoBERTa에 맞게 수정 필요
    num_heads, num_layers = 12, 12

    eval_features, tokenlist = convert_examples_to_features(
        eval_examples, label_list, max_len, tokenizer
    )
    eval_feature = eval_features[0]
    # FIXME: 여러 example 한 번에 처리하려면 수정 필요 (e.g. for로 돌기)

    input_ids = torch.tensor([eval_feature.input_ids], dtype=torch.long).to(device)
    input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long).to(device)
    segment_ids = torch.tensor([eval_feature.segment_ids], dtype=torch.long).to(device)
    label_ids = torch.tensor([eval_feature.label_id], dtype=torch.long).to(device)
    input_len = int(input_mask[0].sum())  

    # example을 모델에 입력해 attention 추출
    model_outputs = model(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        labels=label_ids,
    )
    logits = model_outputs[1]
    attentions = model_outputs[2]       # num_layers * [batch_size, num_heads, seqlen, seqlen]
    pred_label = int(torch.argmax(torch.squeeze(logits, dim=0)))
    # FIXME: 여러 example 한 번에 처리하려면 수정 필요 (squeeze 하지 말고 indexing)

    prob = torch.nn.functional.softmax(logits)
    target_prob = prob[:, label_ids[0]]
    # target_prob: 주어진 example의 정답 label에 해당하는 softmax 값 (모델이 예측한 확률)

    att_all = []
    result_attr = []

    for layer in range(num_layers):
        att_all.append(attentions[layer])
        scaled_att, step = scale_input(
            attentions[layer],
            batch_size,
            num_batches,
        )
        scaled_att.requires_grad_(True)

        attr_all = torch.zeros((batch_size, num_heads, input_len, input_len))
        prob_all = None

        for batch in range(num_batches):
            batch_att = scaled_att[batch*batch_size : (batch+1)*batch_size]

            # one_batch_att: 앞서 얻은 scale_att 중 특정 구간을 추출한 것
            #       수식에서 (k/m)A_h에 해당 (m = num_batches 로 설정한 것으로 보임)
            #       [batch_size, num_heads, seq_len, seq_len]
            
            # attention에 대한 모델 prediction의 gradient 계산
            gradients = torch.autograd.grad(
                outputs=torch.unbind(prob[:, pred_label]),
                inputs=batch_att,
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
        # result_attr: num_layers * [num_heads, input_len, input_len]
    
    if return_attentions:
        return result_attr, att_all
    else:
        return result_attr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--example_index", type=int, required=True)
    parser.add_argument("--num_batches", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda'

    task_name = args.task_name.lower()
    num_labels = num_labels_task[task_name]
    processor = processors[task_name]()

    # Load model and tokenizer
    config = BertConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        output_attentions=True,
    )

    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        state_dict=model_state_dict,
    )
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Generate attribution scores
    attr = generate_attrscore(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        data_dir=args.data_dir,
        example_index=args.example_index,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device,
    )

    # Dump results
    file_name = "attr_ex{0}.json"
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, file_name.format(args.example_index)), 'w') as file:
        for layer_attr in attr:
            output = json.dumps(layer_attr.tolist())
            file.write(output + '\n')

    for i in range(len(attr)):
        print(i)
        print(attr[i].size())
        print()
