import os
import shutil
import sys
import getopt

import pandas as pd
from transformers import AutoTokenizer

from CONSTANTS import *


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hd:m:"

    # Long options
    long_options = ["help", "dataset=", "model="]

    # Prepare flags
    flags = {"dataset": "twitter"}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f"""This script fine tunes the BERTweet tokenizer.\n
        -d or --dataset: dataset name,  available options are {", ".join(DATASET_NAME_OPTIONS)} 
        (default={DATASET_NAME_OPTIONS[0]}).\n
        -m or --model: name of the base tokenizer.
        """)
        sys.exit()

    # checking each argument
    for arg, val in arguments:
        if arg in ("-d", "--dataset"):
            if val in DATASET_NAME_OPTIONS:
                flags["dataset"] = val
            else:
                raise ValueError("Dataset name is not valid.")
        elif arg in ("-m", "--model"):
            flags["model"] = val

    if "model" not in flags.keys():
        print("You must specify a valid model path. Use -m option.")
        sys.exit(1)

    return flags


def list_generator(data: pd.DataFrame):
    for index in range(0, data.shape[0], CHUNK_SIZE):
        # Convert chunk to list and yield
        yield data[SECOND_COLUMN][index: index + CHUNK_SIZE].tolist()


if __name__ == "__main__":
    flags = parsing()
    # Set variables
    path_tokenizer = PATH_TOKENIZER + "-" + flags["dataset"]
    special_tokens = AMAZON_TOKENS if flags["dataset"] == "amazon" else SPECIAL_TOKENS_TWITTER
    path_dataset = PATH_DATASET_AMAZON if flags["dataset"] == "amazon" else PATH_DATASET_TWITTER

    # Delete tokenizer directory if present
    if os.path.exists(path_tokenizer):
        shutil.rmtree(path_tokenizer)

    # Load old version of tokenizer
    tokenizer = AutoTokenizer.from_pretrained(flags["model"])

    # Load dataset as iterator
    data = pd.read_json(open(path_dataset, "r", encoding="utf8"), lines=True)

    new_tokenizer = tokenizer.train_new_from_iterator(list_generator(data),
                                                      vocab_size=max(tokenizer.get_vocab().values()),
                                                      new_special_tokens=special_tokens)

    new_tokenizer.save_pretrained(path_tokenizer)
