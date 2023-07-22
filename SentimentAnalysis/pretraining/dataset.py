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
        self.embedder = Embedder(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())

        # Get token_ids, attention_mask and token_type_ids
        encode_plus_res = self.embedder.encode_plus(text, max_length=self.max_length)
        # Add cls target
        encode_plus_res['cls_targets'] = torch.tensor(self.targets[index], dtype=torch.long)

        # Compute targets for masked tokens
        # tokens_targets = []
        # for i in range(1, len(encode_plus_res['ids'])):
        #     if encode_plus_res['ids'][i] == self.tokenizer.eos_token_id:
        #         break
        #     if encode_plus_res['mask'][i] == 0:
        #         tokens_targets.append(encode_plus_res['ids'][i])

        # Add tokens targets
        # encode_plus_res['tokens_targets'] = torch.nn.functional.pad(
        #     torch.tensor(tokens_targets, dtype=torch.long), (0, PAD_LENGTH - len(tokens_targets)), value=0
        # )

        return encode_plus_res


class TwitterDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        encode_plus_res = self.tokenizer.encode_plus(text,
                                                     None,
                                                     add_special_tokens=True,
                                                     padding='max_length',
                                                     max_length=MAX_LENGTH,
                                                     return_attention_mask=True,
                                                     truncation=True)

        return {
            'ids': torch.tensor(encode_plus_res['input_ids'], dtype=torch.long),
            'mask': torch.tensor(encode_plus_res['attention_mask'], dtype=torch.long),
            'cls_targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
