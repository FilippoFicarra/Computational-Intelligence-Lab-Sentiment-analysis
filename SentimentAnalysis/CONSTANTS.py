CLASSES_NUM = 2
HIDDEN_SIZE = 768
DROPOUT_PROB = 0.1
MODEL = "vinai/bertweet-base"

MODEL_NAME_OPTIONS = ("sparsemax", "robertaMask")
DATASET_NAME_OPTIONS = ("amazon", "twitter")
TOKENIZER_OPTIONS = ("base", "custom")

PATH_POLARITY_AMAZON = "data/sentiment-knowledge/amazon-polarity.csv"
PATH_POLARITY_TWITTER = "data/sentiment-knowledge/twitter-polarity.csv"
PATH_DATASET_AMAZON = "data/datasets/amazon-two-classes.json"
PATH_DATASET_TWITTER = "data/datasets/twitter.json"
PATH_TOKENIZER = "model/tokenizer"

EPOCHS = 20
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 0.00005
PATIENCE = 5

CHUNK_SIZE = 100000
SECOND_COLUMN = "text"

MAX_LENGTH = 128
FRACTION_MASKING = 0.15
LOWER_LIMIT_FRACTION_MASKING = 0.1

VERBOSE_PARAM = 50

AMAZON_OPTIONS = {
    "path": PATH_DATASET_AMAZON,
    "dtype": "label int, text string",
    "grouping_key": "overall",
    "train_fraction": 0.5,
    "eval_fraction": 0.1,
    "columns": ["label", "text"]
}

TWITTER_OPTIONS = {
    "path": PATH_DATASET_TWITTER,
    "dtype": "label int, text string",
    "grouping_key": "label",
    "train_fraction": 0.46,
    "eval_fraction": 0.08,
    "columns": ["label", "text"]
}

SPECIAL_TOKENS_AMAZON = {"additional_special_tokens": ["[EMAIL]", "[URL]", "[XML]", "[PATH]", "[NUMBER]", "[CUR]",
                                                       "[BAD]"]}
