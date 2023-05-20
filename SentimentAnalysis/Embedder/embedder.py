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
# from NGrams.ngrams import NGrams
# import environ
# from  config import Config

class Embedder:
    """
    This class is used to get the embeddings of a text using the BERT model.

    Methods:
        - get_embeddings(text : list[str]) -> torch.Tensor
    """
    def __init__(self, model_name : str = 'bert'):
        self.model_name = model_name
        self.batch_size = 32
        if self.model_name == 'bert':
            self._model_name = 'bert-large-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(self._model_name)
            self.model = BertModel.from_pretrained(self._model_name)
        elif self.model_name == 'roberta':
            self._model_name = 'roberta-large'
            self.tokenizer = RobertaTokenizer.from_pretrained(self._model_name)
            self.model = RobertaModel.from_pretrained(self._model_name)
        self.model.eval()



    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        This function gets the embeddings of a list of texts using the 'model_name' model.

        Args:
            - texts : List[str]
        Returns:
            - embeddings : torch.Tensor
        """

        # Obtain the output embeddings
        embeddings = []
        with torch.no_grad(), tqdm(total=len(texts), desc='Embedding Progress') as pbar:
            for i in range(0, len(texts), self.batch_size):
                
                # Tokenize the input texts
                batch_inputs = self.tokenizer.batch_encode_plus(texts[i:i + self.batch_size], add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
                batch_input_ids = batch_inputs['input_ids']
                batch_attention_masks = batch_inputs['attention_mask']

                if self.model_name == 'bert':
                    batch_token_type_ids = batch_inputs['token_type_ids']

                if self.model_name == 'bert':
                    outputs = self.model(batch_input_ids, attention_mask=batch_attention_masks, token_type_ids=batch_token_type_ids)
                elif self.model_name == 'roberta':
                    outputs = self.model(batch_input_ids, attention_mask=batch_attention_masks)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings for the [CLS] token of each input

                embeddings.append(batch_embeddings)
                pbar.update(batch_input_ids.shape[0])

        embeddings = torch.cat(embeddings, dim=0)  # Concatenate embeddings from all batches
        return embeddings

