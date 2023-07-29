import torch
import torch.nn as nn

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModel
from .dataframe_manager import DataFrameManager
import numpy as np
from tqdm import tqdm
import pandas as pd

from typing import List, Optional, Tuple, Union
from sparsemax import Sparsemax
import math

class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.sparsemax = Sparsemax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.sparsemax(attention_scores)

        # let the mean be 0

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



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


DATASET_COLUMNS = ["text", "label"]
DATASET_ENCODING = "ISO-8859-1"


class SMARTRobertaClassificationModel(nn.Module):
    
    def __init__(self, model, weight = 0.02):
        super().__init__()
        self.model = model 
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):

        embed = self.model.model.embeddings(input_ids) 

        def eval(embed):
            outputs = self.model.model(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:,0,:]
            logits = self.model.classifier(pooled) 
            return logits 
        
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        state = eval(embed)

        loss = nn.functional.cross_entropy(state, labels)
        loss += self.weight * smart_loss_fn(embed, state)
        
        return state, loss

    def forward_eval(self, input_ids, attention_mask, labels):
        
        embed = self.model.model.embeddings(input_ids) 
        
        def eval(embed):
            outputs = self.model.model(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:,0,:]
            logits = self.model.classifier(pooled) 
            return logits 
       
        state = eval(embed)
        loss = nn.functional.cross_entropy(state, labels)
        return state, loss 

config = {
    "models" : {
        "bertweet_1" : AutoModel.from_pretrained("vinai/bertweet-base"),
        "bertweet_2" : AutoModel.from_pretrained("vinai/bertweet-base"),
        "bertweet_3" : AutoModel.from_pretrained("vinai/bertweet-base")
    },
    "tokenizer" : AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False),
}

for j in range(1,4):
    for i in range(2):
        config["models"]["bertweet_" + str(j)].encoder.layer[i-1].attention.self = RobertaSelfAttention(config=config["models"]["bertweet_" + str(j)].config)


encode_map = {"NEGATIVE" : 0, "POSITIVE" : 1}


def _setup():
    len_train = 100000
    len_test = 15000

    dataFrameManage = DataFrameManager()
    train_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/train_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
    test_df = dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
    positive_train_df = train_df[train_df["target"]== "POSITIVE"]
    negative_train_df = train_df[train_df["target"]== "NEGATIVE"]

    positive_test_df = test_df[test_df["target"] == "POSITIVE"]
    negative_test_df = test_df[test_df["target"] == "NEGATIVE"]

    train_dataset = {
        "df_1" : pd.concat([positive_train_df[:len_train], negative_train_df[:len_train]], axis=0).sample(frac=1, random_state=42),
        "df_2" : pd.concat([positive_train_df[len_train : 2*len_train], negative_train_df[len_train : 2*len_train]], axis=0).sample(frac=1, random_state=42),
        "df_3" : pd.concat([positive_train_df[2*len_train : 3*len_train], negative_train_df[2*len_train : 3*len_train]], axis=0).sample(frac=1, random_state=42)
    }
    test_dataset = pd.concat([positive_test_df[:len_test], negative_test_df[:len_test]], axis=0).sample(frac=1, random_state=42)

    return train_dataset, test_dataset

def train():
    
    train_dataset, test_dataset = _setup()
    
    tokenizer = config["tokenizer"]
    num_models = 3

    for j in  tqdm(range(1,num_models + 1)):
        print(f"Model {j}")

        bertweet = SMARTRobertaClassificationModel(Classifier(model=config["models"][f"bertweet_{j}"], num_classes=2))

        train_texts, eval_texts = train_dataset[f"df_{j}"]["text"].to_list(), test_dataset["text"].to_list()
        train_lbls, eval_lbls = train_dataset[f"df_{j}"]["target"].map(encode_map).to_list(), test_dataset["target"].map(encode_map).to_list()

        # Tokenize the texts
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

        # Convert the dataset to PyTorch tensors
        train_set = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings["input_ids"]),
            torch.tensor(train_encodings["attention_mask"]),
            torch.tensor(train_lbls)
        )
        eval_set = torch.utils.data.TensorDataset(
            torch.tensor(eval_encodings["input_ids"]),
            torch.tensor(eval_encodings["attention_mask"]),
            torch.tensor(eval_lbls)
        )

        # Define the optimizer and learning rate
        model = bertweet
        optimizer = AdamW(model.parameters(), lr=1e-5)
        

        # Fine-tuning loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        model.train()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=64)

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
            
            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels )
                
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                
                predicted_train_labels = torch.argmax(logits, dim=1)

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

                    predicted_eval_labels = torch.argmax(logits, dim=1)
                    
                    correct_eval += (predicted_eval_labels == eval_labels).sum().item()
                    
            epoch_eval_loss = eval_loss / eval_total
            eval_losses.append(epoch_eval_loss)
            print(f"Epoch {epoch+1} Eval Loss: {epoch_eval_loss}")

            epoch_eval_accuracy = correct_eval / eval_total
            eval_accuracies.append(epoch_eval_accuracy)
            print(f"Epoch {epoch+1} Eval accuracy : {epoch_eval_accuracy}")
            
            if epoch_eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = epoch_eval_accuracy
                best_model_path = f"best_model_{j}"
                model.model.save_model(f'smart_bertweet_{j}.pt')
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
    test_dataset =  dataFrameManage.load_dataframe(filepath="data/twitter-datasets/preprocessed/test_data_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False, test = True)
    
    eval_texts, eval_lbls = test_dataset["text"].to_list(), test_dataset["target"].map(encode_map).to_list()

    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)
    eval_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(eval_encodings["input_ids"]),
                    torch.tensor(eval_encodings["attention_mask"]),
                    torch.tensor(eval_lbls)
                )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
    tokenizer = config["tokenizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    eval_loss = 0.0
    eval_total = 0
    predicted = []
    for j in  range(1,4):
        model = Classifier(config["models"][f"bertweet_{j}"],2)
        model.load_model(f"smart_bertweet_{j}.pt")
        model.to(device)
        temp = []
        with torch.no_grad():
            for eval_batch in eval_loader:
                eval_input_ids, eval_attention_mask, eval_labels = eval_batch
                eval_input_ids = eval_input_ids.to(device)
                eval_attention_mask = eval_attention_mask.to(device)
                eval_labels = eval_labels.to(device)
        
                loss, logits = model(eval_input_ids, attention_mask=eval_attention_mask, labels = eval_labels)
                eval_loss += loss.item()
                eval_total += eval_labels.size(0)
        
                # logits = eval_outputs.logits
                predicted_eval_labels = torch.argmax(logits, dim=1)
                temp.append(predicted_eval_labels)
            print("accuracy",(torch.cat(temp, dim=0) == torch.tensor(eval_lbls,device=device)).sum().item()/len(torch.cat(temp, dim=0)))
            predicted.append(torch.cat(temp, dim=0).cpu().numpy())

    results = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predicted)
    print("Majority accuracy",(torch.from_numpy(results).to(device) == torch.tensor(eval_lbls,device=device)).sum().item()/len(results))
    
if __name__ == "__main__":
    train()
    eval()
    