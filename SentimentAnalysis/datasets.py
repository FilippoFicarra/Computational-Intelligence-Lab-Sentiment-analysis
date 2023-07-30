import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, CLIPProcessor

from CONSTANTS import *
from masker import Masker


class ReviewDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.text = dataframe.reviewText
        self.targets = dataframe.overall
        self.max_length = max_length
        self.masker = Masker(tokenizer, twitter=False)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())

        # Get token_ids, attention_mask and token_type_ids
        encode_plus_res = self.masker.encode_plus(text, max_length=self.max_length)
        # Add cls target
        encode_plus_res['cls_targets'] = torch.tensor(self.targets[index], dtype=torch.long)

        return encode_plus_res


class TwitterDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH,
                 use_embedder=False):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_length = max_length
        self.use_masker = use_embedder
        self.masker = Masker(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        if self.use_masker:
            encode_plus_res = self.masker.encode_plus(text, max_length=self.max_length)
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
            'cls_targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


class TwitterDatasetTest(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH,
                 use_embedder=False):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.max_length = max_length
        self.use_masker = use_embedder
        self.masker = Masker(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        if self.use_masker:
            encode_plus_res = self.masker.encode_plus(text, max_length=self.max_length)
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
        }


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.text = dataframe.text
        self.label = dataframe.label
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        res = self.tokenizer(text, padding='longest', truncation=True)
        return {
            'input_ids': torch.nn.functional.pad(torch.tensor(res["input_ids"], dtype=torch.long),
                                                 (0, TOKENIZER_SIZE - len(torch.tensor(res["input_ids"]))),
                                                  mode="constant", value=self.pad_token_id),
            'attention_mask': torch.nn.functional.pad(torch.tensor(res["attention_mask"], dtype=torch.long),
                                                      (0, - len(torch.tensor(res["attention_mask"]))),
                                                      mode="constant", value=0),
            'cls_targets': torch.tensor(self.label[index], dtype=torch.long)
        }


class TwitterDatasetEnsamble(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_length = max_length
        self.masker = Masker(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        encode_plus_res_masker = self.masker.encode_plus(text, max_length=self.max_length)

        encode_plus_res = self.tokenizer.encode_plus(text,
                                                     None,
                                                     add_special_tokens=True,
                                                     padding='max_length',
                                                     max_length=self.max_length,
                                                     return_attention_mask=True,
                                                     truncation=True)

        return {
            'input_ids': torch.tensor(encode_plus_res['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encode_plus_res['attention_mask'], dtype=torch.long),
            'input_ids_masker': encode_plus_res_masker['input_ids'],
            'attention_mask_masker': encode_plus_res_masker['attention_mask'],
            'cls_targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


class TwitterDatasetEnsambleTest(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.max_length = max_length
        self.masker = Masker(tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        encode_plus_res_masker = self.masker.encode_plus(text, max_length=self.max_length)

        encode_plus_res = self.tokenizer.encode_plus(text,
                                                     None,
                                                     add_special_tokens=True,
                                                     padding='max_length',
                                                     max_length=self.max_length,
                                                     return_attention_mask=True,
                                                     truncation=True)

        return {
            'input_ids': torch.tensor(encode_plus_res['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encode_plus_res['attention_mask'], dtype=torch.long),
            'input_ids_masker': encode_plus_res_masker['input_ids'],
            'attention_mask_masker': encode_plus_res_masker['attention_mask']
        }
