import torch

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
        self.epoch = 0

    def forward(self, input_ids, attention_mask):
        base_model_output = self.base_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            return_dict=True)
        # Get cls output probabilities
        hidden_state = base_model_output['last_hidden_state']
        cls_first_linear = self.first_linear(hidden_state[:, 0])
        cls_dropout = self.classifier_dropout(cls_first_linear)
        cls_output = self.second_linear(cls_dropout)

        return cls_output

    def update_epoch(self, epoch):
        self.epoch = epoch

    def model_representation(self):
        return {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict()
        }

    def load_model(self, file_path):
        state = torch.load(file_path)
        self.epoch = state['epoch']
        self.load_state_dict(state['model_state_dict'])
