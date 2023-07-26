import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from CONSTANTS import *
from masker import Embedder


class ReviewDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.text = dataframe.reviewText
        self.targets = dataframe.overall
        self.max_length = max_length
        self.embedder = Embedder(tokenizer, twitter=False)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())

        # Get token_ids, attention_mask and token_type_ids
        encode_plus_res = self.embedder.encode_plus(text, max_length=self.max_length)
        # Add cls target
        encode_plus_res['cls_targets'] = torch.tensor(self.targets[index], dtype=torch.long)

        return encode_plus_res


class DatasetPreprocessingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH,
                 use_embedder=False):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.labels = dataframe.label
        self.max_length = max_length
        self.use_embedder = use_embedder
        self.embedder = Embedder(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text.iloc[index].split())
        if self.use_embedder:
            encode_plus_res = self.embedder.encode_plus(text, max_length=self.max_length)
            input_ids = encode_plus_res['input_ids']
            attention_mask = encode_plus_res['attention_mask']
        else:
            encode_plus_res = self.tokenizer.encode_plus(text,
                                                         None,
                                                         add_special_tokens=True,
                                                         padding='max_length',
                                                         max_length=self.max_length,
                                                         return_attention_mask=True,
                                                         truncation=True)
            input_ids = torch.tensor(encode_plus_res['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encode_plus_res['attention_mask'], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cls_targets': torch.tensor(self.labels.iloc[index], dtype=torch.long)
        }


class TwitterDataset(Dataset):
    def __init__(self, data: dict, device):
        self.ids = data['input_ids'].to(device)
        self.masks = data['attention_mask'].to(device)
        self.targets = data["cls_targets"].to(device)

    def __len__(self):
        return self.targets.shape(0)

    def __getitem__(self, index):
        return {
            'input_ids': self.ids[index],
            'attention_mask': self.masks[index],
            'cls_targets': self.targets[index]
        }
