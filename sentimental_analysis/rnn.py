
import torch.nn as nn

import torch
import torch.nn as nn
from tqdm import tqdm
from dataframe_manager import DataFrameManager
from transformers import AutoTokenizer
import torchtext

import torch.optim as optim

DATASET_COLUMNS = ["text", "label"]
DATASET_ENCODING = "ISO-8859-1"


class LSTMNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        
        super(LSTMNet,self).__init__()
        
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        
        # LSTM layer process the vector sequences 
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )
        
        # Dense layer to predict 
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)
        
        # Thanks to packing, LSTM don't see padding tokens 
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        
        packed_output, (hidden_state,cell_state) = self.lstm(packed_embedded)
        
        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.sigmoid(dense_outputs)
        
        return outputs
    
# We'll use this helper to compute accuracy
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def _setup():

    dataFrameManage = DataFrameManager()
    train_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/train_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
    test_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)

    return train_df, test_df

def main():
    
    encode_map = {"NEGATIVE" : "emotsad", "POSITIVE" : "emothappy"}
    
    train_df, test_df = _setup()
    train_df["target"] = train_df["target"].map(encode_map)
    test_df["target"] = test_df["target"].map(encode_map)
    train_df.to_csv("data/twitter-datasets/preprocessed/train_preprocessed_rnn.csv", index=False)
    test_df.to_csv("data/twitter-datasets/preprocessed/test_preprocessed_rnn.csv", index=False)
    
    # train_df["text"] = train_df["text"] + " " + train_df["target"].map(encode_map)
    # test_df["text"] = test_df["text"] + " " + test_df["target"].map(encode_map)
    
    # Define the fields for the dataset
    TEXT = torchtext.data.Field(sequential=True, tokenize='spacy', lower=True, tokenizer_language="en_core_web_sm")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=True, tokenizer_language="en_core_web_sm")

    # Load the dataset
    data_fields = [('text', TEXT), ('label', LABEL)]
    train_data, test_data = torchtext.data.TabularDataset.splits(
        path='data/twitter-datasets/preprocessed/',
        train='train_preprocessed_rnn.csv',
        validation='test_preprocessed_rnn.csv',
        format='csv',
        fields=data_fields,
    )

    # Build the vocabulary
    TEXT.build_vocab(train_data, min_freq=2)
    LABEL.build_vocab(train_data)

    # Create iterators for batching with bucketing
    BATCH_SIZE = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device {}.".format(device))
    print("Batch size: {}".format(BATCH_SIZE))
    
    train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,  # Sort within each bucket
        device=device
    )
   
    SIZE_OF_VOCAB = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    NUM_HIDDEN_NODES = 64
    NUM_OUTPUT_NODES = 1
    NUM_LAYERS = 2
    BIDIRECTION = True
    DROPOUT = 0.2 
    
    torch.manual_seed(0)
    model =  LSTMNet(SIZE_OF_VOCAB,
                EMBEDDING_DIM,
                NUM_HIDDEN_NODES,
                NUM_OUTPUT_NODES,
                NUM_LAYERS,
                BIDIRECTION,
                DROPOUT
               )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    
    EPOCH_NUMBER = 15
    
    for epoch in tqdm(range(1,EPOCH_NUMBER+1)):
        
        train_loss, train_acc = train(model, train_iterator,optimizer,criterion)
        valid_loss, valid_acc = evaluate(model, test_iterator,criterion)
        
        # Showing statistics
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print()

        

def train(model,iterator,optimizer,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    for batch in tqdm(iterator):
        
        # cleaning the cache of optimizer
        optimizer.zero_grad()
        
        print(batch.text)
        text = batch.text
        text_lengths = text.shape[1]
        
        # forward propagation and squeezing
        predictions = model(text,text_lengths).squeeze()
        
        # computing loss / backward propagation
        loss = criterion(predictions,batch.type)
        loss.backward()
        
        # accuracy
        acc = binary_accuracy(predictions,batch.type)
        
        # updating params
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # deactivate the dropouts
    model.eval()
    
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in  tqdm(iterator):
            text = batch.text
            text_lengths = text.shape[1]
        
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch.type)
            acc = binary_accuracy(predictions, batch.type)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":
    main()
    