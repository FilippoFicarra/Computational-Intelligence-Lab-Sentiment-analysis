import torch.nn as nn


class SMARTRobertaClassificationModel(nn.Module):

    def __init__(self, model, weight=0.02):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, input_ids, attention_mask):
        return self.model(inputs_ids=input_ids, attention_mask=attention_mask)
