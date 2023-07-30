import gc
import math
import os
import random

import pandas as pd
from transformers import AutoModel

from bert_tweet_sparsemax import BertTweetWithSparsemax, RobertaSelfAttention
from bert_tweet_with_mask import BertTweetWithMask
from clip_with_classification import CLIPWithClassificationHead
from CONSTANTS import *


def get_training_and_validation_dataframes(path, dtype, grouping_key, train_fraction, eval_fraction, columns):
    # Load dataframe with dataset

    # Set random seed so the same dataset is sampled every time
    random_seed = 42
    random.seed(random_seed)

    df = pd.read_json(path, lines=True, dtype=dtype)
    dataset_size = df.shape[0]

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

    # Sample dataset for validation
    eval_values = []
    for key, group in grouped_df_after_drop.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(eval_fraction * len(group))):
            eval_values.append((key, df_after_drop.iloc[ind, 1]))

    # Delete unused variable
    del df, grouped_df, df_after_drop, grouped_df_after_drop
    gc.collect()

    # Return training and validation dataframes plus the splits
    return pd.DataFrame(data=training_values, columns=columns), pd.DataFrame(data=eval_values, columns=columns), \
        len(training_values) / dataset_size, len(eval_values) / dataset_size


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def get_model(pt_file, device):
    file_path = os.path.join(PATH_MODELS, pt_file)
    mask = False
    # Do something with each .pt file
    if "mask" in file_path.lower():
        model = BertTweetWithMask(AutoModel.from_pretrained(MODEL))
        mask = True
    else:
        if "last-2-sparsemax" in pt_file:
            model = BertTweetWithSparsemax(AutoModel.from_pretrained(MODEL))
            model.base_model.encoder.layer[-1].attention.self = RobertaSelfAttention(config=model.base_model.config)
            model.base_model.encoder.layer[-2].attention.self = RobertaSelfAttention(config=model.base_model.config)
        elif "no-sparsemax" in pt_file:
            model = BertTweetWithSparsemax(AutoModel.from_pretrained(MODEL))
        elif "sparsemax-first-2-last-2" in pt_file:
            model = BertTweetWithSparsemax(AutoModel.from_pretrained(MODEL))
            model.base_model.encoder.layer[0].attention.self = RobertaSelfAttention(config=model.base_model.config)
            model.base_model.encoder.layer[1].attention.self = RobertaSelfAttention(config=model.base_model.config)
            model.base_model.encoder.layer[-2].attention.self = RobertaSelfAttention(config=model.base_model.config)
            model.base_model.encoder.layer[-1].attention.self = RobertaSelfAttention(config=model.base_model.config)
        elif "clip" in pt_file:
            model = CLIPWithClassificationHead()
        else:
            raise Exception("Model not yet supported")

    model.load_model(file_path)
    freeze_model_parameters(model)
    model.to(device)
    return model, mask
