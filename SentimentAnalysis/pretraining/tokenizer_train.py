from transformers import AutoTokenizer
from transformers import RobertaTokenizerFast
import pandas as pd
import shutil
import os

# GLOBAL VARIABLES

PATH_DATASET = "data/datasets/dataset-cleaned-no-unknown.json"
PATH_TOKENIZER = "model/tokenizer"
CHUNK_SIZE = 100000
SECOND_COLUMN_DATASET = "reviewText"


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
    data = pd.read_json(open(PATH_DATASET, "r", encoding="utf8"), lines=True)

    new_tokenizer = tokenizer.train_new_from_iterator(list_generator(data), vocab_size=40000,
                                                      new_special_tokens=["[EMAIL]", "[URL]", "[XML]", "[PATH]",
                                                                          "[NUMBER]", "[USD]", "[EUR]", "[GBP]",
                                                                          "[JPY]", "[INR]", "[BAD]", "[UNKNOWN]"])

    new_tokenizer.save_pretrained(PATH_TOKENIZER)
