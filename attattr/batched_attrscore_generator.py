import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging

import torch

from .model_input import ModelInput
from .attrscore_generator import AttrScoreGenerator

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BatchedAttrScoreGenerator(AttrScoreGenerator):
    """
    fine-tuned model을 준비해두었다가, 모델의 입력이 주어지면 그에 대한
    self-attention attribution score를 산출함.
    이때 모델에 입력 문장을 하나씩 전달하지 않고 batch 단위로 묶어서 처리함.
    """

    def generate_attrscore(self, inputs: ModelInput):
        """
        input을 받아서 그에 대한 self-attention attribution score를 산출함.
        Returns: `num_layers * [(num_heads, input_len, input_len)]`
        """

        input_batch_size = inputs.input_ids.size(0)

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
        attentions = model_outputs[2]       # num_layers * [input_batch_size, num_heads, maxlen, maxlen]
        pred_label = int(torch.argmax(torch.squeeze(logits, dim=0)))

        input_batch_gradients = []
        result_attr = []

        for layer in range(num_layers):

            for input_batch_idx in range(input_batch_size):
                scaled_att, step = self.__scale_input(attentions[layer][input_batch_idx])
                scaled_att.requires_grad_(True)

                sum_gradients = torch.zeros((num_heads, padded_len, padded_len)).to(self.device)

                for batch in range(self.num_batches):
                    batch_att = scaled_att[batch*self.batch_size : (batch+1)*self.batch_size]
                    # batch_att: 앞서 얻은 scale_att 중 특정 구간을 추출한 것
                    #       수식에서 (k/m)A_h에 해당
                    #       [batch_size, num_heads, maxlen, maxlen]
                    
                    # attention에 대한 모델 prediction의 gradient 계산
                    gradients, = self.model(
                        input_ids=inputs.input_ids,
                        token_type_ids=inputs.token_type_ids,
                        attention_mask=inputs.attention_mask,
                        replace_attention=batch_att,
                        pred_label=pred_label,
                        tar_layer=layer,
                    )

                    gradients = torch.sum(gradients, dim=0)     # batch 차원을 따라 sum
                    sum_gradients = torch.add(sum_gradients, gradients)
                    # sum_gradients: gradients의 누적합 [num_heads, maxlen, maxlen]
                
                input_batch_gradients.append(sum_gradients.data)

            # gradient에 attention을 elementwise 곱함
            input_batch_gradients = torch.stack(input_batch_gradients, dim=0)
            input_batch_gradients = input_batch_gradients[:, :, :input_len, :input_len] * step[0, :input_len, :input_len]
            result_attr.append(input_batch_gradients)
            
        # result_attr: num_layer * [input_batch_size, num_heads, input_len, input_len]
        return result_attr
