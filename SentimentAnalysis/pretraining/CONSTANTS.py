VOCABULARY_SIZE = 50265
CLASSES_NUM = 2
HIDDEN_SIZE = 768
DROPOUT_PROB = 0.1
MODEL = "vinai/bertweet-base"

MODEL_NAME_OPTIONS = ("sparsemax", "robertaMask")
DATASET_NAME_OPTIONS = ("amazon", "twitter")

PATH_POLARITY = "data/sentiment-knowledge/polarity.csv"
PATH_DATASET_AMAZON = "data/datasets/dataset-two-classes.json"
PATH_DATASET_TWITTER = "data/datasets/twitter-dataset.json"
PATH_TOKENIZER = "model/tokenizer"
MODEL_PATHNAME = "tweet_competition"

EPOCHS = 20
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 0.00001
NUMBER_OF_TPU_WORKERS = 8
PATIENCE = 5

CHUNK_SIZE = 100000
SECOND_COLUMN_DATASET = "reviewText"

MAX_LENGTH = 128
FRACTION_MASKING = 0.15
LOWER_LIMIT_FRACTION_MASKING = 0.1

VERBOSE_PARAM = 50

AMAZON_OPTIONS = {
    "path": PATH_DATASET_AMAZON,
    "dtype": "overall int, reviewText string",
    "grouping_key": "overall",
    "train_fraction": 0.5,
    "eval_fraction": 0.1,
    "columns": ["overall", "reviewText"]
}

TWITTER_OPTIONS = {
    "path": PATH_DATASET_TWITTER,
    "dtype": "label int, text string",
    "grouping_key": "label",
    "train_fraction": 0.25,
    "eval_fraction": 1 / 15,
    "columns": ["label", "text"]
}

SPECIAL_TOKENS_AMAZON = {"additional_special_tokens": ["[EMAIL]", "[URL]", "[XML]", "[PATH]", "[NUMBER]", "[CUR]",
                                                       "[BAD]", "<user>", "<url>"]}
SPECIAL_TOKENS_TWITTER = {"additional_special_tokens": ["[CUR]", "[BAD]", "<user>", "<url>"]}
