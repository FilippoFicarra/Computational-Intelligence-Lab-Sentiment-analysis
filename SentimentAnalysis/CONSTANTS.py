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
PATH_DATASET_TWITTER_TEST = "data/datasets/twitter-test.json"
PATH_TOKENIZER = "tokenizer"
PATH_MODELS = "model"
PATH_PREDICTIONS = "predictions"
PATH_MAJORITY = "majority"
PATH_LOSSES_AND_ACCURACIES = "measures"

EPOCHS = 20
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 1e-5
PATIENCE = 1

CHUNK_SIZE = 100000
SECOND_COLUMN = "text"

MAX_LENGTH = 64
FRACTION_MASKING = 0.15
LOWER_LIMIT_FRACTION_MASKING = 0.1

VERBOSE_PARAM = 100
VERBOSE_PARAM_FOR_SAVING = 100

MASK_TOKEN_FOR_REPLACEMENT = "<mask>"
TWITTER_THRESHOLD = 5.
AMAZON_THRESHOLD = 25.

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
    "train_fraction": 0.8,
    "eval_fraction": 1,
    "columns": ["label", "text"]
}

DTYPE_TWITTER_TEST = "text string"
TEST_BATCH_SIZE = 16

AMAZON_TOKENS = {"additional_special_tokens": ["[EMAIL]", "[URL]", "[XML]", "[PATH]", "[NUMBER]", "[CUR]", "[BAD]"]}

# Variables for plotting
LABEL_TRAINING_ACCURACY = "Training Accuracy"
LABEL_VAL_ACCURACY = "Validation Accuracy"

LABEL_TRAINING_LOSS = "Training Loss"
LABEL_VAL_LOSS = "Validation Loss"

TITLE_TRAINING_ACCURACY = "Training Accuracy"
TITLE_VAL_ACCURACY = "Validation Accuracy"
TITLE_TRAINING_LOSS = "Training Loss"
TITLE_VAL_LOSS = "Validation Loss"

Y_LABEL_ACCURACY = "Accuracy"
Y_LABEL_LOSS = "Loss"
X_LABEL = "Epoch"
