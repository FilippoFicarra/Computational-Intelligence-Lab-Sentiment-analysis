import torch.nn as nn
import torch

from smart_pytorch.smart_pytorch import SMARTLoss
from smart_pytorch.loss import *


class SMARTRobertaClassificationModel(nn.Module):

    def __init__(self, model, weight=0.02):
        super().__init__()
        self.model = model
        self.weight = weight
        self.epoch = 0

    def forward(self, input_ids, attention_mask, labels):
        embed = self.model.base_model.embeddings(input_ids)

        def eval_fn(embed):
            outputs = self.model.base_model(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state
            logits = self.model.classifier(pooled)
            return logits

        smart_loss_fn = SMARTLoss(eval_fn=eval_fn, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
        state = eval_fn(embed)
        gt_state = nn.functional.one_hot(labels, num_classes=2)
        loss = nn.functional.cross_entropy(state, gt_state)
        loss += self.weight * smart_loss_fn(embed, state)

        return state, loss

    def forward_eval(self, input_ids, attention_mask, labels):
        embed = self.model.base_model.embeddings(input_ids)

        def eval_fn(embed):
            outputs = self.model.base_model(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state
            logits = self.model.classifier(pooled)
            return logits

        state = eval_fn(embed)
        loss = nn.functional.cross_entropy(state, labels)
        return state, loss

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
