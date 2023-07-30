import torch
from transformers import CLIPModel
from CONSTANTS import *


class CLIPWithClassificationHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP)
        self.linear = torch.nn.Linear(CLIP_SIZE, CLASSES_NUM)
        self.epoch = 0

    def forward(self, input_ids, attention_mask):
        text_features = self.clip.get_text_features(input_ids, attention_mask)
        return self.linear(text_features)

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

