from DataFrameManager.dataframeManager import DataFrameManager
import os
import torch
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm
from NGrams.ngrams import NGrams

class Embedder:
    """
    This class is used to get the embeddings of a text using the BERT model.

    Methods:
        - get_embeddings(text : str) -> torch.Tensor
    """
    def __init__(self):
        self.bert_model_name = 'bert-large-uncased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_model.eval()
        self.roberta_model_name = 'roberta-large'
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(self.roberta_model_name)
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_model_name)
        self.roberta_model.eval()

    def get_embeddings(self, text : str, model_name = 'bert') -> torch.Tensor:
        """
        This function gets the embeddings of a text using the  'model_name' model.

        Args:
            - text : str
            - model_name : str
        Returns:
            - embeddings : torch.Tensor
        """
        if model_name == 'bert':
            tokenizer = self.bert_tokenizer
            model = self.bert_model
        elif model_name == 'roberta':
            tokenizer = self.roberta_tokenizer
            model = self.roberta_model
            
        # Tokenize the text
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # Obtain the output embeddings
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            embeddings = outputs[0][0]  # take the first element of the batch and the last hidden layer
        return embeddings


## Example usage
if __name__ == '__main__':
    directory_path = 'SentimentAnalysis/Data'
    filename = 'preprocessed.csv'
    file_path = os.path.join(directory_path, filename)
    data_frame_manager = DataFrameManager()
    N = 4

    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    if os.path.exists(file_path):
        print("File already exists, loading it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/preprocessed.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS, preprocess=False)
        print("File loaded.")
        print(len(df))
    else:
        print("File does not exist, loading the original dataset and preprocessing it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/training.1600000.processed.noemoticon.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS).dropna(subset=['text'])
        data_frame_manager.export_dataframe(df, filepath="SentimentAnalysis/Data/preprocessed.csv")
        print("Preprocessing done and saved to CSV file.")
    
    train_df, test_df = data_frame_manager.split(df = df)

    embedder = Embedder()

    # Get the embeddings for the test set
    if not os.path.exists('SentimentAnalysis/Data/test_embeddings.npy'):
        test_embeddings = test_df.text.progress_apply(embedder.get_embeddings)
        print("File does not exist, saving the embeddings...")
        with open('SentimentAnalysis/Data/test_embeddings.npy', 'xb') as f:
            np.save(f, test_embeddings.to_numpy())
    else:
        print("File already exists, loading it...")
        test_embeddings = np.load('SentimentAnalysis/Data/test_embeddings.npy', allow_pickle=True)
        print(f"Loaded {test_embeddings.shape} array from 'SentimentAnalysis/Data/test_embeddings.npy'")

    # Get the embeddings for the train set
    if not os.path.exists('SentimentAnalysis/Data/train_embeddings.npy'):
        train_embeddings = train_df.text.progress_apply(embedder.get_embeddings)
        print("File does not exist, saving the embeddings...")
        with open('SentimentAnalysis/Data/train_embeddings.npy', 'xb') as f:
            np.save(f, train_embeddings.to_numpy())
    else:
        print("File already exists, loading it...")
        train_embeddings = np.load('SentimentAnalysis/Data/train_embeddings.npy', allow_pickle=True)
        print(f"Loaded {train_embeddings.shape} array from 'SentimentAnalysis/Data/train_embeddings.npy'")

    # n_grams = NGrams()
    
    # print("Getting the n-grams for the test set...")
    # test_ngrams = [n_grams.generate_ngrams(embeddings, N) for embeddings in tqdm(test_embeddings, desc='Generating n-grams')]
    # print("Getting the n-grams for the train set...")
    # train_ngrams = [n_grams.generate_ngrams(embeddings, N) for embeddings in tqdm(train_embeddings, desc='Generating n-grams')]

    
    

