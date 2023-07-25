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


