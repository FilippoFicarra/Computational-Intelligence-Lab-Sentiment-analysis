import os
import shutil

import pandas as pd
from transformers import AutoTokenizer

from CONSTANTS import *


def list_generator(data: pd.DataFrame):
    for index in range(0, data.shape[0], CHUNK_SIZE):
        # Convert chunk to list and yield
        yield data[SECOND_COLUMN_DATASET][index: index + CHUNK_SIZE].tolist()


if __name__ == "__main__":
    # Delete tokenizer directory if present
    if os.path.exists(PATH_TOKENIZER):
        shutil.rmtree(PATH_TOKENIZER)

    # Load old version of tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load dataset as iterator
    data = pd.read_json(open(PATH_DATASET_AMAZON, "r", encoding="utf8"), lines=True)

    new_tokenizer = tokenizer.train_new_from_iterator(list_generator(data), vocab_size=VOCABULARY_SIZE,
                                                      new_special_tokens=SPECIAL_TOKENS_AMAZON)

    new_tokenizer.save_pretrained(PATH_TOKENIZER)
