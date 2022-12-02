
class ModelInput:
    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
    ):
        self.input_ids = input_ids.long()
        self.token_type_ids = token_type_ids.long() if token_type_ids else None
        self.attention_mask = attention_mask.long()
        self.labels = labels.long()
        self.input_len = int(attention_mask[0].sum())
