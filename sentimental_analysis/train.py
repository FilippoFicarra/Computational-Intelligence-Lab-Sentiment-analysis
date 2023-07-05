from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from data_frame_manager.data_frame_manager import DataFrameManager
import numpy as np
from tqdm import tqdm
# import xgboost as xgb
import pandas as pd


DATASET_COLUMNS = ["text", "label"]
DATASET_ENCODING = "ISO-8859-1"


class SMARTRobertaClassificationModel(nn.Module):
    
    def __init__(self, model, weight = 0.02):
        super().__init__()
        self.model = model 
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):

        embed = self.model.roberta.embeddings(input_ids) 

        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state
            # print(self.model.classifier)
            logits = self.model.classifier(pooled) 
            return logits 
        
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        state = eval(embed)
        # print(state)
        # print(state.shape, labels.shape)
        # print(state.view(-1, 2).shape, labels.view(-1).shape)
        loss = nn.functional.cross_entropy(state, labels)
        loss += self.weight * smart_loss_fn(embed, state)
        
        return state, loss
    
def main():
    
    dataFrameManage = DataFrameManager()
    train_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/train_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
    test_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)

    encode_map = {"NEGATIVE" : 0, "POSITIVE" : 2}

    train_labels = train_df["target"].map(encode_map).to_list()
    test_labels = test_df["target"].map(encode_map).to_list()

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    model = SMARTRobertaClassificationModel(model)

    # Split the dataset into training and evaluation sets
    train_texts, eval_texts, train_lbls, eval_lbls = train_df["text"][:100000].to_list(), test_df["text"][:20000].to_list(), train_labels[:100000], test_labels[:20000]

    # Tokenize the texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)


    # Convert the dataset to PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings["input_ids"]),
        torch.tensor(train_encodings["attention_mask"]),
        torch.tensor(train_lbls)
    )
    eval_dataset = torch.utils.data.TensorDataset(
        torch.tensor(eval_encodings["input_ids"]),
        torch.tensor(eval_encodings["attention_mask"]),
        torch.tensor(eval_lbls)
    )

    # Define the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Fine-tuning loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=24)

    train_losses = []
    eval_losses = []

    train_accuracies = []
    eval_accuracies = []

    for epoch in tqdm(range(3)):  # Adjust the number of epochs as needed
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        model.train() # theta hat
        
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # print(input_ids.shape, attention_mask.shape, labels.shape)
            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels )
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
            l = torch.cat((logits[:, 0:1], logits[:, 2:]), dim=1)
            predicted_train_labels = torch.where(torch.argmax(l, dim=1) == 1, torch.tensor([2], device = device), torch.tensor([0], device = device))

            correct_train += (predicted_train_labels == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_accuracy = correct_train / total_train
        print(f"Epoch {epoch+1} Train accuracy : {epoch_train_accuracy}")
        train_accuracies.append(epoch_train_accuracy)
            
        # Calculate and store training loss for the epoch
        epoch_train_loss = running_train_loss / total_train
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss}")

        # Calculate and store validation loss for the epoch
        model.eval()
        eval_loss = 0.0
        
        eval_total = 0
        correct_eval = 0

        with torch.no_grad():
            for eval_batch in eval_loader:
                eval_input_ids, eval_attention_mask, eval_labels = eval_batch
                eval_input_ids = eval_input_ids.to(device)
                eval_attention_mask = eval_attention_mask.to(device)
                eval_labels = eval_labels.to(device)

                eval_logits, eval_loss = model(eval_input_ids, attention_mask=eval_attention_mask, labels = eval_labels)
                eval_loss += eval_loss.item()
                eval_total += eval_labels.size(0)

                logits = eval_logits
                l = torch.cat((logits[:, 0:1], logits[:, 2:]), dim=1)
                predicted_eval_labels = torch.where(torch.argmax(l, dim=1) == 1, torch.tensor([2], device = device), torch.tensor([0], device = device))
                
                correct_eval += (predicted_eval_labels == eval_labels).sum().item()
                
        epoch_eval_loss = eval_loss / eval_total
        eval_losses.append(epoch_eval_loss)

        print(f"Epoch {epoch+1} Eval Loss: {epoch_eval_loss}")

        epoch_eval_accuracy = correct_eval / eval_total
        print(f"Epoch {epoch+1} Eval accuracy : {epoch_eval_accuracy}")

        eval_accuracies.append(epoch_eval_accuracy)

        model.train()
        
    model.eval()
    model.save_pretrained("twitter-roberta-base-sentiment-latest-BCELoss")


if __name__ == "__main__":
    main()