from DataFrameManager.dataframeManager import DataFrameManager
import os
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import tqdm.auto as tqdm

class Embedder:
    """
    This class is used to get the embeddings of a text using the BERT model.
    Functions:
        - get_embeddings(text : str) -> torch.Tensor
    """
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()

    def get_embeddings(self, text : str) -> torch.Tensor:
        # Tokenize the text
        input_ids = torch.tensor([self.tokenizer.encode_plus(text, add_special_tokens=True)['input_ids']])
        attention_masks = torch.tensor([self.tokenizer.encode_plus(text, add_special_tokens=True)['attention_mask']])
        token_type_ids = torch.tensor([self.tokenizer.encode_plus(text, add_special_tokens=True)['token_type_ids']])

        # Obtain the output embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            embeddings = outputs[0][0]  # take the first element of the batch and the last hidden layer
        return embeddings


## Example usage
if __name__ == '__main__':
    directory_path = 'SentimentAnalysis/Data'
    filename = 'preprocessed.csv'
    file_path = os.path.join(directory_path, filename)
    data_frame_manager = DataFrameManager()

    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    if os.path.exists(file_path):
        print("File already exists, loading it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/preprocessed.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS, preprocess=False)
        print("File loaded.")
        print(len(df))
    else:
        print("File does not exist, loading the original dataset and preprocessing it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/training.1600000.processed.noemoticon.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS).dropna()
        data_frame_manager.export_dataframe(df, filepath="SentimentAnalysis/Data/preprocessed.csv")
        print("Preprocessing done and saved to CSV file.")
    
    train_df, test_df = data_frame_manager.split(df)

    embedder = Embedder()
    # Get the embeddings for the first train set
    embeddings = train_df.text.progress_apply(embedder.get_embeddings)
    print(embeddings)
    with open('Data/embeddings.npy', 'wb') as f:
        np.save('Data/embeddings.npy', embeddings.to_numpy())

