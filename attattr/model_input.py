class ModelInput:
    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
        device='cuda',
    ):
        self.input_ids = input_ids.to(device)
        self.token_type_ids = token_type_ids.to(device) if token_type_ids else None
        self.attention_mask = attention_mask.to(device)
        self.labels = labels.to(device)
        self.input_len = int(attention_mask[0].sum())
