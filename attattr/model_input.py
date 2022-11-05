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
