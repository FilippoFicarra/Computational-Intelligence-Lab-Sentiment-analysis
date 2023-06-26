from functools import partial
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from tqdm import tqdm
from typing import List
from common import utils
import multiprocessing
import pandas as pd
from tqdm.auto import tqdm
from torch.nn import DataParallel
import time
import os

class Embedder:
    """
    This class is used to get the embeddings of a text using the BERT model.

    Methods:
        - get_embeddings(text : list[str]) -> torch.Tensor
    """
    def __init__(self, model_name : str = 'bert'):
        self.model_name = model_name
        self.batch_size = 64
        if 'bert' in self.model_name and not 'roberta' in self.model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
        elif 'roberta' in self.model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaModel.from_pretrained(self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps") # for m1 mac
        self.model.to(self.device)
        self.model = DataParallel(self.model)
        self.model.eval()



    def get_embeddings(self, texts: List[str], np_file: str) -> None:
        """
        This function gets the embeddings of a list of texts using the 'model_name' model.

        Args:
            - texts : List[str]
            - np_file : str
        Returns:
            - None
        """

        if not os.path.isfile(np_file):
            with open(np_file, 'wb'):
                pass
        embeddings = []
        with torch.no_grad(), tqdm(total=len(texts), desc='Embedding Progress') as pbar:
            for i in range(0, len(texts), self.batch_size):
                # Tokenize the input texts
                batch_inputs = self.tokenizer.batch_encode_plus(texts[i:i + self.batch_size], add_special_tokens=True, return_tensors='pt', padding=True, truncation=True).to(self.device)
                batch_input_ids = batch_inputs['input_ids']
                batch_attention_masks = batch_inputs['attention_mask']

                if 'bert' in self.model_name and not 'roberta' in self.model_name:
                    batch_token_type_ids = batch_inputs['token_type_ids']

                if 'bert' in self.model_name and not 'roberta' in self.model_name:
                    outputs = self.model(batch_input_ids, attention_mask=batch_attention_masks, token_type_ids=batch_token_type_ids)
                elif 'roberta' in self.model_name:
                    outputs = self.model(batch_input_ids, attention_mask=batch_attention_masks)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings for the [CLS] token of each input
                embeddings.append(batch_embeddings)
                if i == 0:
                    np.save(np_file, torch.cat(embeddings, dim=0).cpu().numpy())
                    embeddings = []
                elif i % 1000 == 0:  # Save every 32 batches
                    existing_embeddings = np.load(np_file, allow_pickle=True)
                    new_embeddings = np.concatenate((existing_embeddings, torch.cat(embeddings, dim=0).cpu().numpy()), axis=0)
                    np.save(np_file, new_embeddings)
                    embeddings = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                pbar.update(batch_input_ids.shape[0])

                del batch_inputs, batch_input_ids, batch_attention_masks, outputs, batch_embeddings
                if 'bert' in self.model_name and not 'roberta' in self.model_name:
                    del batch_token_type_ids
        # Save any remaining embeddings in the last array
        if len(embeddings) > 0:
            existing_embeddings = np.load(np_file, allow_pickle=True)
            new_embeddings = np.concatenate((existing_embeddings, torch.cat(embeddings, dim=0).cpu().numpy()), axis=0)
            np.save(np_file, new_embeddings)

