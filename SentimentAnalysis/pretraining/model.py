import torch
import numpy as np
from transformers import RobertaModel
from CONSTANTS import *


class Roberta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.token_classifier_linear = torch.nn.Linear(HIDDEN_SIZE, VOCABULARY_SIZE)
        # self.token_softmax = torch.nn.Softmax()
        self.pre_review_classifier_linear = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.pre_review_classifier_dropout = torch.nn.Dropout(p=0.1)
        self.review_classifier_linear = torch.nn.Linear(HIDDEN_SIZE, OVERALL_NUMBER)
        # self.review_softmax = torch.nn.Softmax()

    def forward(self, input_ids, attention_mask, token_type_ids, eos_token_id):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      return_dict=True)
        # Get cls output probabilities
        hidden_state = roberta_output['last_hidden_state']
        cls_first_linear = self.pre_review_classifier_linear(hidden_state[:, 0])
        cls_dropout = self.pre_review_classifier_dropout(cls_first_linear)
        cls_second_linear = self.review_classifier_linear(cls_dropout)
        # output_cls = self.review_softmax(cls_second_linear)

        # Get probabilities for masked tokens
        tokens_values_batch = []
        for i in range(hidden_state.size(0)):
            tokens_values = []
            for j in range(1, MAX_LENGTH):
                if input_ids[i, j] == eos_token_id:
                    break
                if attention_mask[i, j] == 0:
                    token_linear = self.token_classifier_linear(hidden_state[i, j, :])
                    tokens_values.append(token_linear)
            # Add padding to get tensors all the same size
            padding = torch.full((PAD_LENGTH - len(tokens_values), VOCABULARY_SIZE), fill_value=-np.inf)
            for ind in range(padding.size(0)):
                padding[ind, 0] = 1.0
            tokens_values_batch.append(torch.cat((torch.stack(tokens_values), padding), dim=0))

        return {'cls': cls_second_linear, 'tokens': torch.stack(tokens_values_batch)}
