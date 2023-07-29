from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from dataframe_manager import DataFrameManager
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
            logits = self.model.classifier(pooled) 
            return logits 
        
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        state = eval(embed)
        loss = nn.functional.cross_entropy(state, labels)
        loss += self.weight * smart_loss_fn(embed, state)
        
        return state, loss

    def forward_eval(self, input_ids, attention_mask, labels):
        
        embed = self.model.roberta.embeddings(input_ids) 
        
        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state
            logits = self.model.classifier(pooled) 
            return logits 
       
        state = eval(embed)
        loss = nn.functional.cross_entropy(state, labels)
        return state, loss 
        


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

config = {
    "models" : {
        "bertweet_1" : AutoModelForSequenceClassification.from_pretrained(MODEL),
        "bertweet_2" :  AutoModelForSequenceClassification.from_pretrained(MODEL),
        "bertweet_3" :  AutoModelForSequenceClassification.from_pretrained(MODEL)
    },

}

def train():
    
    dataFrameManage = DataFrameManager()
    train_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/train_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
    test_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)

    encode_map = {"NEGATIVE" : 0, "POSITIVE" : 2}

    train_labels = train_df["target"].map(encode_map).to_list()
    test_labels = test_df["target"].map(encode_map).to_list()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    
    num_models = 3
    for j in  tqdm(range(1,num_models + 1)):
        print(f"Model {j}")
        model = SMARTRobertaClassificationModel(config["models"][f"bertweet_{j}"])

        # Split the dataset into training and evaluation sets
        train_texts, eval_texts, train_lbls, eval_lbls = train_df["text"][:500000].to_list(), test_df["text"][:20000*5].to_list(), train_labels[:100000*5], test_labels[:20000*5]

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
        print(f"Using device: {device}")
        model.to(device)
        model.train()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)

        train_losses = []
        eval_losses = []

        train_accuracies = []
        eval_accuracies = []
        
        best_eval_accuracy = float("-inf")
        best_model_path = None
        
        patience = 2  # Number of epochs to wait for improvement
        no_improvement_count = 0
        
        for epoch in tqdm(range(10)):  # Adjust the number of epochs as needed
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            model.train() # theta hat
            
            for step, batch in enumerate(tqdm(train_loader)):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
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

                    logits,  loss  = model.forward_eval(eval_input_ids, attention_mask=eval_attention_mask, labels = eval_labels)
                    eval_loss += loss.item()
                    eval_total += eval_labels.size(0)

                    l = torch.cat((logits[:, 0:1], logits[:, 2:]), dim=1)
                    predicted_eval_labels = torch.where(torch.argmax(l, dim=1) == 1, torch.tensor([2], device = device), torch.tensor([0], device = device))
                    
                    correct_eval += (predicted_eval_labels == eval_labels).sum().item()
                    
            epoch_eval_loss = eval_loss / eval_total
            eval_losses.append(epoch_eval_loss)
            print(f"Epoch {epoch+1} Eval Loss: {epoch_eval_loss}")

            epoch_eval_accuracy = correct_eval / eval_total
            eval_accuracies.append(epoch_eval_accuracy)
            print(f"Epoch {epoch+1} Eval accuracy : {epoch_eval_accuracy}")
            
            if epoch_eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = epoch_eval_accuracy
                best_model_path = f"smart_bertweet_{j}.pt"
                model.model.save_pretrained(best_model_path)
                no_improvement_count = 0  # Reset the counter
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break      

            model.train()
            
        model.eval()

def eval():
    dataFrameManage = DataFrameManager()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    val_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_data_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False, test = True)
    eval_encodings = tokenizer(val_df["text"].to_list(), truncation=True, padding=True)
    eval_dataset = torch.utils.data.TensorDataset(
        torch.tensor(eval_encodings["input_ids"]),
        torch.tensor(eval_encodings["attention_mask"])
    )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
    print(len(val_df))
    preds = []
    for j in  range(1,4):
        model =  config["models"][f"bertweet_{j}"]
        model.load_model(f"smart_bertweet_{j}.pt")
        model.to(device)
        model.eval()

        temp = []
        
        with torch.no_grad():

            for batch in tqdm(eval_loader):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                predicted_labels =  torch.argmax(logits, dim=1)
                temp.append(predicted_labels)
            preds.append(torch.cat(temp, dim=0).cpu().numpy())


    preds_array = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=preds)
    mapped_array = np.where(preds_array == 0, -1, 1)
    predict_df = pd.DataFrame(mapped_array, columns=["Prediction"], index=pd.Index(range(1, len(mapped_array)+1), name="Id"))

    predict_df.to_csv(f"data/twitter-datasets/predictions_smart_bertweet_majority_100k.csv", index=True, index_label="Id")
    # print("End of predicion")


if __name__ == "__main__":
    train()
    eval()