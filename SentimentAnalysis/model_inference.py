"""
This module is designed to perform inference on the twitter test set only. The precondition is that the models are
named such that if a model requires sentiment-based masking, its name should contain 'mask', while all other models
should have a name which does not contain the word 'mask'.
"""

import sys
import getopt
import os

from CONSTANTS import *

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import TwitterDatasetTest


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hm:f:"

    # Long options
    long_options = ["help", "models=", "filename="]

    # Discover model names by looking at model folder
    # Get the list of file names in the folder
    with os.scandir(PATH_MODELS) as entries:
        file_names = [entry.name for entry in entries if entry.is_file()]

    # Prepare flags
    flags = {"models": file_names}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f'This script trains a model on a TPU with multiple cores.\n\
        -m or --models: list of model filenames. Model filenames must be separated by commas without spaces.'
              + f'Available options are {", ".join(file_names)} (default={",".join(file_names)}).\n\
        -f or --filename: name of the file for the predictions (valid name with no extension).')
        sys.exit()

    # checking each argument
    for arg, val in arguments:
        if arg in ("-m", "--model"):
            flags["model"] = val.split(",")
            for model_name in flags["model"]:
                if model_name not in file_names:
                    raise ValueError("Model argument not valid.")
        elif arg in ("-n", "--filename"):
            flags["filename"] = '{}.csv'.format(val)

        if "filename" not in flags.keys():
            flags["filename"] = 'submission.csv'

    return flags


if __name__ == "__main__":
    # Get test dataframe
    df = pd.read_json(PATH_DATASET_TWITTER_TEST, lines=True, dtype=DTYPE_TWITTER_TEST)

    # Create tokeinizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create dataset and data loader with and without special masking
    dataset_no_special_mask = TwitterDatasetTest()

    # get
    #
    # tokenizer = config["tokenizer"]
    # len_test = 20000
    #
    # eval_texts, eval_lbls = test_df["text"][:len_test].to_list(), test_labels[:len_test]
    # print(len(eval_texts))
    # eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)
    # eval_dataset = torch.utils.data.TensorDataset(
    #     torch.tensor(eval_encodings["input_ids"]),
    #     torch.tensor(eval_encodings["attention_mask"]),
    #     torch.tensor(eval_lbls)
    # )
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16)
    # tokenizer = config["tokenizer"]
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    #
    # predicted = []
    # for j in range(1, 4):
    #     model = Classifier(AutoModel.from_pretrained("vinai/bertweet-base"), 2)
    #     model.load_model(f"bertweet_400k_sparsemax_{j}.pt")
    #     model.to(device)
    #     temp = []
    #     eval_loss = 0.0
    #
    #     eval_total = 0
    #     correct_eval = 0
    #     with torch.no_grad():
    #         for eval_batch in eval_loader:
    #             eval_input_ids, eval_attention_mask, eval_labels = eval_batch
    #             eval_input_ids = eval_input_ids.to(device)
    #             eval_attention_mask = eval_attention_mask.to(device)
    #             eval_labels = eval_labels.to(device)
    #
    #             loss, logits = model(eval_input_ids, attention_mask=eval_attention_mask, labels=eval_labels)
    #             eval_loss += loss.item()
    #             eval_total += eval_labels.size(0)
    #
    #             # logits = eval_outputs.logits
    #             predicted_eval_labels = torch.argmax(logits, dim=1)
    #             temp.append(predicted_eval_labels)
    #         predicted.append(torch.cat(temp, dim=0).cpu().numpy())
    #
    #
    # def majority_vote(predicted):
    #     return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predicted)
    #
    #
    # def weighted_majority_vote(predictions, weights):
    #     weighted_votes = [predictions[i] * weights[i] for i in range(len(predictions))]
    #
    #     aggregated_votes = np.sum(weighted_votes, axis=0)
    #
    #     final_prediction = np.where(aggregated_votes >= (sum(weights) / 2), 1, 0)
    #
    #     return final_prediction
    #
    #
    # preds = predicted
    # majority = majority_vote(predicted)
    # print("Majority ",
    #       (torch.from_numpy(majority).to(device) == torch.tensor(eval_lbls, device=device)).sum().item() / len(majority))
    #
    # weighted_majority = weighted_majority_vote(preds, [0.4, 0.3, 0.3])
    # print("Weighted majority ",
    #       (torch.from_numpy(weighted_majority).to(device) == torch.tensor(eval_lbls, device=device)).sum().item() / len(
    #           weighted_majority))
    #
    # dataFrameManage = DataFrameManager()
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    #
    # tokenizer = config["tokenizer"]
    #
    # val_df = dataFrameManage.load_dataframe(filepath="./data/twitter-datasets/preprocessed/test_data_preprocessed.csv",
    #                                         encoding=DATASET_ENCODING, preprocess=False, test=True)
    # eval_encodings = tokenizer(val_df["text"].to_list(), truncation=True, padding=True)
    # eval_dataset = torch.utils.data.TensorDataset(
    #     torch.tensor(eval_encodings["input_ids"]),
    #     torch.tensor(eval_encodings["attention_mask"])
    # )
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
    # print(len(val_df))
    # preds = []
    # for j in range(1, 4):
    #     model = Classifier(config["models"][f"bertweet_{j}"], 2)
    #     model.load_model(f"bertweet_400k_sparsemax_{j}.pt")
    #     model.to(device)
    #     model.eval()
    #
    #     temp = []
    #
    #     with torch.no_grad():
    #
    #         for batch in tqdm(eval_loader):
    #             input_ids, attention_mask = batch
    #             input_ids = input_ids.to(device)
    #             attention_mask = attention_mask.to(device)
    #
    #             logits = model(input_ids, attention_mask=attention_mask)
    #             predicted_labels = torch.argmax(logits, dim=1)
    #             temp.append(predicted_labels)
    #         preds.append(torch.cat(temp, dim=0).cpu().numpy())
    #
    # preds_array = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=preds)
    # print(len(preds_array))
    # mapped_array = np.where(preds_array == 0, -1, 1)
    # predict_df = pd.DataFrame(mapped_array, columns=["Prediction"],
    #                           index=pd.Index(range(1, len(mapped_array) + 1), name="Id"))
    #
    # predict_df.to_csv(f"../SentimentAnalysis/data/twitter-datasets/bertweet_400k_sparsemax_majority.csv", index=True,
    #                   index_label="Id")
    # print("End of predicion")
