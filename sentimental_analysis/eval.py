import torch
import torch.nn as nn

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModel
from dataframe_manager import DataFrameManager
import numpy as np
from tqdm import tqdm
# import xgboost as xgb
import pandas as pd

class Classifier(nn.Module):
    def __init__(self, model, num_classes, hidden_size=768):
        super(Classifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.epoch = 0

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        # cls = self.dropout(cls)
        logits = self.classifier(cls)

        if labels is not None:
            loss = self.loss_function(logits, labels)
            return loss, logits
        else:
            return logits

    def save_model(self, file_path):
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
        }
        torch.save(state, file_path)

    def load_model(self, file_path):
        state = torch.load(file_path)
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model_state_dict'])
        self.classifier.load_state_dict(state['classifier_state_dict'])

config = {
    "models" : {
        "bertweet_1" : AutoModel.from_pretrained("vinai/bertweet-base"),
        "bertweet_2" : AutoModel.from_pretrained("vinai/bertweet-base"),
        "bertweet_3" : AutoModel.from_pretrained("vinai/bertweet-base")
    },
    "tokenizer" : AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False),
}

DATASET_COLUMNS = ["text", "label"]
DATASET_ENCODING = "ISO-8859-1"


dataFrameManage = DataFrameManager()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = config["tokenizer"]

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
    model = Classifier(config["models"][f"bertweet_{j}"],2)
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
print(len(preds_array))
mapped_array = np.where(preds_array == 0, -1, 1)
predict_df = pd.DataFrame(mapped_array, columns=["Prediction"], index=pd.Index(range(1, len(mapped_array)+1), name="Id"))

predict_df.to_csv(f"data/twitter-datasets/predictions_smart_bertweet_majority_100k.csv", index=True, index_label="Id")
# print("End of predicion")