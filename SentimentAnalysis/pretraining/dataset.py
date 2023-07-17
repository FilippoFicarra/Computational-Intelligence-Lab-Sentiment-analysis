import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import pandas as pd
from masker import Embedder


class ReviewDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=512):
        self.tokenizer = tokenizer
        self.text = dataframe.reviewText
        self.targets = dataframe.overall
        self.max_length = max_length
        self.embedder = Embedder(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())

        # Get token_ids, attention_mask and token_type_ids
        encode_plus_res = self.embedder.encode_plus(text, max_length=self.max_length)

        # Add target
        encode_plus_res['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return encode_plus_res
