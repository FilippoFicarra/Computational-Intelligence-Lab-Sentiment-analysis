import gc
import math
import random

import pandas as pd
import torch
from transformers import AutoTokenizer

from CONSTANTS import *
from dataset import TwitterDataset


def get_training_and_validation_dataframes(path, dtype, grouping_key, train_fraction, eval_fraction, columns):
    # Load dataframe with dataset

    # Set random seed so the same dataset is sampled every time
    random_seed = 42
    random.seed(random_seed)

    df = pd.read_json(path, lines=True, dtype=dtype)

    # Group by overall
    grouped_df = df.groupby(grouping_key)

    # Sample 20 % of dataset for training
    training_values = []
    indeces_to_drop_training = []
    for key, group in grouped_df.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(train_fraction * len(group))):
            training_values.append((key, df.iloc[ind, 1]))
            indeces_to_drop_training.append(ind)

    df_after_drop = df.drop(indeces_to_drop_training).reset_index(drop=True)
    grouped_df_after_drop = df_after_drop.groupby(grouping_key)

    # Sample 1 % of dataset for validation
    eval_values = []
    for key, group in grouped_df_after_drop.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(eval_fraction * len(group))):
            eval_values.append((key, df_after_drop.iloc[ind, 1]))

    # Delete unused variable
    del df, grouped_df, df_after_drop, grouped_df_after_drop
    gc.collect()

    # Return training and validation dataframes
    return pd.DataFrame(data=training_values, columns=columns), pd.DataFrame(data=eval_values, columns=columns)


def save_tensor(df, tokenizer, path):
    # Save tensors for training
    dataset = TwitterDataset(df, tokenizer)
    samples = []
    for sample in iter(dataset):
        try:
            tensor_sample = {key: value for key, value in sample.items()}
            samples.append(tensor_sample)
        except KeyError:
            pass

    stacked_samples_training = {key: torch.stack([sample[key] for sample in samples])
                                for key in samples[0].keys()}
    torch.save(stacked_samples_training, path)


if __name__ == "__main__":
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Get dataframe
    training, evaluation = get_training_and_validation_dataframes(**TWITTER_OPTIONS)

    # Save training
    save_tensor(training, tokenizer, TENSOR_TRAINING_DATA_PATH)

    # Save evaluation
    save_tensor(evaluation, tokenizer, TENSOR_EVAL_DATA_PATH)
