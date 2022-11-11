import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import random
import logging
import json

import numpy as np
import torch

from transformers_cutoff import RobertaConfig

from .modeling_roberta import RobertaForSequenceClassification
from .model_input import ModelInput

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AttrScoreGenerator:
    """
    fine-tuned model을 준비해두었다가, 모델의 입력이 주어지면 그에 대한
    self-attention attribution score를 산출함.
    """

    num_labels_task_map = {
        "cola": 2, "mnli": 3, "mrpc": 2, "rte": 2, "sst-2": 2,
        "qqp": 2, "qnli": 2, "wnli": 2, "sts-b": 1,
    }

    def __init__(
        self,
        model_name,
        task_name,
        model_file=None,
        num_batches=1,
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

        self.__prepare_model()

    def generate_attrscore(self, inputs: ModelInput):
        """
        input을 받아서 그에 대한 self-attention attribution score를 산출함.
        Returns: `num_layers * [(num_heads, input_len, input_len)]`
        """

        assert inputs.input_ids.size(0) == 1, "The input batch size should be 1"
        assert inputs.token_type_ids is None or inputs.token_type_ids.size(0) == 1, "The input batch size should be 1"
        assert inputs.attention_mask.size(0) == 1, "The input batch size should be 1"
        assert inputs.labels.size(0) == 1, "The input batch size should be 1"

        num_layers, num_heads = 12, 12
        input_len = inputs.input_len
        padded_len = inputs.input_ids.size(-1)

        model_outputs = self.model(
            input_ids=inputs.input_ids,
            token_type_ids=inputs.token_type_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.labels,
        )
        logits = model_outputs[1]
        attentions = model_outputs[2]       # num_layers * [1(batch_size), num_heads, seqlen, seqlen]
        pred_label = int(torch.argmax(torch.squeeze(logits, dim=0)))

        result_attr = []

        for layer in range(num_layers):
            scaled_att, step = self.__scale_input(attentions[layer].squeeze(0))
            # scaled_att: 입력된 attribution matrix와 baseline의 차이를 계산해서, batch_size*num_batch(=:m)로
            #       나눈 뒤 거기에 다시 1부터 N까지의 수를 차례로 곱해서 그 결과들을 cat으로 연결한 것.
            #       default가 "baseline = zero tensor"이므로, (1/m)A부터 (m/m)A=A까지를 연결해놓은 것에 해당.
            #       [m, num_heads, seq_len, seq_len]
            # step: 앞에서 계산한 차이를 N으로 나눈 것. baseline이 zero일 때 (1/m)A에 해당.
            scaled_att.requires_grad_(True)

            sum_gradients = torch.zeros((num_heads, padded_len, padded_len)).to(self.device)

            # 원래 총 m(= num_batches * batch_size)번의 계산을 해야 하지만 batch 단위로 묶어서 수행
            for batch in range(self.num_batches):
                batch_att = scaled_att[batch*self.batch_size : (batch+1)*self.batch_size]
                # batch_att: 앞서 얻은 scale_att 중 특정 구간을 추출한 것
                #       수식에서 (k/m)A_h에 해당
                #       [batch_size, num_heads, seq_len, seq_len]
                
                # attention에 대한 모델 prediction의 gradient 계산
                gradients, = self.model(
                    input_ids=inputs.input_ids,
                    token_type_ids=inputs.token_type_ids,
                    attention_mask=inputs.attention_mask,
                    # labels=inputs.labels,
                    replace_attention=batch_att,
                    pred_label=pred_label,
                    tar_layer=layer,
                )
                # gradients: 모델이 예측한 label에 해당하는 softmax 값의, batch_att에 대한 gradient
                #       [batch_size, num_heads, seq_len, seq_len]
                #       TODO: 왜 gradient를 A_h가 아니라 (k/m)A에 대해 계산할까?

                gradients = torch.sum(gradients, dim=0)     # batch 차원을 따라 sum
                sum_gradients = torch.add(sum_gradients, gradients)
                # sum_gradients: gradients의 누적합 [num_heads, seq_len, seq_len]

            # gradient에 attention을 곱함
            sum_gradients = sum_gradients[:, :input_len, :input_len] * step[0, :input_len, :input_len]
            result_attr.append(sum_gradients.data)
            
        # result_attr: num_layer * [num_heads, input_len, input_len]
        return result_attr

    def __prepare_model(self):
        """
        로컬 디렉토리에 저장된 model checkpoint를 가져와 모델을 사용할 수 있도록 준비함.
        """

        # Load model config
        config = RobertaConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels_task_map[self.task_name],
            output_attentions=True,     # 이 옵션을 주어야 attention을 얻을 수 있음
        )

        # Load model checkpoint from the specified local directory
        if self.model_file is not None:
            state_dict = torch.load(self.model_file)
        else:
            state_dict = None
        
        # Load model
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            state_dict=state_dict,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model for AttAttr is prepared.")

    def __scale_input(self, attention):
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

    @classmethod
    def dump_attr(cls, attr, output_dir):
        """
        (For test) 산출된 attribution score를 json 파일로 저장함.
        """

        file_name = "attr.json"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, file_name), 'w') as file:
            for layer_attr in attr:
                output = json.dumps(layer_attr.tolist())
                file.write(output + '\n')

        # Print tensor sizes
        for i in range(len(attr)):
            print(i)
            print(attr[i].size())
            print()
