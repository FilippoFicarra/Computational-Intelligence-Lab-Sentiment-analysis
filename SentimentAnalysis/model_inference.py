"""
This module is designed to perform inference on the twitter test set only. The precondition is that the models are
named such that if a model requires sentiment-based masking, its name should contain 'mask', while all other models
should have a name which does not contain the word 'mask'. The module computes predictions for all models saved in the
folder 'model', and saves the predictions in the folder 'predictions'.
"""

import os
import csv

from CONSTANTS import *

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import TwitterDatasetTest
from bert_tweet_sparsemax import BertTweetWithSparsemax
from bert_tweet_with_mask import BertTweetWithMask


if __name__ == "__main__":
    # Get test dataframe
    df = pd.read_json(PATH_DATASET_TWITTER_TEST, lines=True, dtype=DTYPE_TWITTER_TEST)

    # Create tokeinizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create dataset and data loader with and without special masking
    dataset_no_special_mask = TwitterDatasetTest(df, tokenizer)
    dataset_special_mask = TwitterDatasetTest(df, tokenizer, use_embedder=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for filename in os.listdir(PATH_MODELS):
        file_path = os.path.join(PATH_MODELS, filename)

        # Check if the current item is a file (not a subdirectory)
        if os.path.isfile(file_path) and ".pt" in file_path:
            # Load model
            checkpoint = torch.load(file_path)
            model_state_dict = checkpoint["model_state_dict"]
            if "mask" not in filename.lower():
                model = BertTweetWithSparsemax(AutoModel.from_pretrained(MODEL))
                model.load_model(file_path)
                model.to(device)
                dataset = dataset_no_special_mask
            else:
                model = BertTweetWithMask(AutoModel.from_pretrained(MODEL))
                model.load_model(file_path)
                model.to(device)
                dataset = dataset_special_mask

            # Create dataloader
            loader = DataLoader(dataset,
                                batch_size=TEST_BATCH_SIZE,
                                shuffle=False)
            # Define list of predictions
            all_predictions = []
            # Compute predictions
            with torch.no_grad():
                for sample in iter(loader):
                    ids = sample['input_ids'].to(device)
                    mask = sample['attention_mask'].to(device)
                    predictions = model(ids, mask)

                    all_predictions.append(torch.argmax(predictions, dim=1))

            # Produce final tensor with predictions
            all_predictions = torch.cat(all_predictions, dim=0)

            # Save predictions to file
            data = [(idx + 1, value.item()) for idx, value in enumerate(all_predictions)]

            # Check if the directory for predictions already exists
            if not os.path.exists(PATH_PREDICTIONS):
                # If it doesn't exist, create the directory
                os.makedirs(PATH_PREDICTIONS)

            # Write the data to a CSV file
            with open(PATH_PREDICTIONS + "/" + filename.replace(".pt", "") + ".csv", "w", newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Id', 'Prediction'])
                csv_writer.writerows(data)
