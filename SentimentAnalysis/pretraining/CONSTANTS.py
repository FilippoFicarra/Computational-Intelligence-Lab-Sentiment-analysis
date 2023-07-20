import math

VOCABULARY_SIZE = 50265
OVERALL_NUMBER = 5
HIDDEN_SIZE = 768

PATH_POLARITY = "data/sentiment-knowledge/polarity.csv"
PATH_DATASET = "data/datasets/dataset-cleaned-no-unknown.json"
PATH_TOKENIZER = "model/tokenizer"
PATH_MODEL = "model/model/model.pt"

EPOCHS = 10
TRAIN_BATCH_SIZE = 256
LEARNING_RATE = 0.001
WARMUP_STEPS = 4000
NUMBER_OF_TPU_WORKERS = 8

CHUNK_SIZE = 100000
SECOND_COLUMN_DATASET = "reviewText"

MAX_LENGTH = 128
FRACTION_MASKING = 0.15
LOWER_LIMIT_FRACTION_MASKING = 0.1

# PAD_LENGTH = math.ceil(FRACTION_MASKING * MAX_LENGTH)
