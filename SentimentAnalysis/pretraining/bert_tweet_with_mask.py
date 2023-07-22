import torch
from torch_xla.utils.serialization import load
# import numpy as np

from CONSTANTS import *


class BertTweetWithMask(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        # self.token_classifier_linear = torch.nn.Linear(HIDDEN_SIZE, VOCABULARY_SIZE)
        # self.token_softmax = torch.nn.Softmax()
        self.first_linear = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.classifier_dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.second_linear = torch.nn.Linear(HIDDEN_SIZE, CLASSES_NUM)
        self.sigmoid = torch.nn.Sigmoid()
        self.epoch = 0

    def forward(self, input_ids, attention_mask):
        base_model_output = self.base_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            return_dict=True)
        # Get cls output probabilities
        hidden_state = base_model_output['last_hidden_state']
        cls_first_linear = self.first_linear(hidden_state[:, 0])
        cls_dropout = self.classifier_dropout(cls_first_linear)
        cls_second_linear = self.second_linear(cls_dropout)
        cls_output = self.sigmoid(cls_second_linear)

        return cls_output

        # Get probabilities for masked tokens
        # tokens_values_batch = []
        # for i in range(hidden_state.size(0)):
        #     tokens_values = []
        #     for j in range(1, MAX_LENGTH):
        #         if input_ids[i, j] == eos_token_id:
        #             break
        #         if attention_mask[i, j] == 0:
        #             token_linear = self.token_classifier_linear(hidden_state[i, j, :])
        #             tokens_values.append(token_linear)
        #     # Add padding to get tensors all the same size
        #     padding = torch.full((PAD_LENGTH - len(tokens_values), VOCABULARY_SIZE), fill_value=-np.inf)
        #     for ind in range(padding.size(0)):
        #         padding[ind, 0] = 1.0
        #     tokens_values_batch.append(torch.cat((torch.stack(tokens_values), padding), dim=0))
        # return {'cls': cls_second_linear, 'tokens': torch.stack(tokens_values_batch)}

    def update_epoch(self, epoch):
        self.epoch = epoch

    def model_representation(self):
        return {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict()
        }

    def load_model(self, file_path):
        state = load(file_path)
        self.epoch = state['epoch']
        self.load_state_dict(state['model_state_dict'])
