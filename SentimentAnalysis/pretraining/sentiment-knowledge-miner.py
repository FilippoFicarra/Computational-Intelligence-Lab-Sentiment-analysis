import pandas as pd
from utility_functions import *
from concurrent.futures import ThreadPoolExecutor, wait
import threading
import math

# GLOBAL VARIABLES
PATH_OCCURRENCES = "data/sentiment-knowledge/all-words-with-occurrences.csv"
PATH_SEEDS_POSITIVE = "data/sentiment-knowledge/seeds_positive.txt"
PATH_SEEDS_NEGATIVE = "data/sentiment-knowledge/seeds_negative.txt"
PATH_PMI = "data/pmi.csv"
CHUNK_SIZE = 100000
PATH_DATASET = "data/dataset-cleaned-no-unknown.jsonl"
FIRST_COLUMN_OCC = "word"
SECOND_COLUMN_OCC = "occurrences"
SECOND_COLUMN_DATASET = "reviewText"


def interval_generator(start, end, interval_size):
    current = start
    while current < end:
        yield [current, min(current + interval_size - 1, end)]
        current += interval_size


def co_occurrences_calc(chunk, locks_vec, df_cooccurences, seeds):
    # Iterate over objects
    for ind in range(0, chunk.shape[0]):
        # Convert string to lowercase
        string = chunk.iloc[ind, 1]
        # Clean and tokenize
        tokens = tokenize_with_sequences(remove_symbols_before_tokenization(string))
        # Iterate over tokens
        for i in range(0, len(tokens)):
            # Check if current token is a seed
            if tokens[i] in seeds:
                if i - 1 >= 0 and tokens[i - 1] in df_cooccurences.index:
                    # First, acquire lock on cell
                    row_number = df_cooccurences.index.get_loc(tokens[i - 1])
                    locks_vec[row_number].acquire()
                    try:
                        # Increase count of co-occurrences
                        df_cooccurences.at[tokens[i - 1], tokens[i]] += 1
                    finally:
                        # Release lock
                        locks_vec[row_number].release()
                if i + 1 < len(tokens) and tokens[i + 1] in df_cooccurences.index:
                    # First, acquire lock on cell
                    row_number = df_cooccurences.index.get_loc(tokens[i + 1])
                    locks_vec[row_number].acquire()
                    try:
                        # Increase count of co-occurrences
                        df_cooccurences.at[tokens[i + 1], "_" + tokens[i]] += 1
                    finally:
                        # Release lock
                        locks_vec[row_number].release()


def pmi(c_w1_w2, c_w1, c_w2, N):
    # Calculate pmi
    return max(math.log((c_w1_w2 * N) / (c_w1 * c_w2)), 0)


def pmi_calc(df_cooccurences: pd.DataFrame, df_words, row_range, N):
    # Iterate over rows in row_range
    for i in range(row_range[0], row_range[1]):
        # Iterate over columns
        for j in range(0, df_cooccurences.shape[1]):
            # Compute pmi for current element
            df_cooccurences.iloc[i, j] = pmi(df_cooccurences.iloc[i, j],
                                             df_words.at[df_cooccurences.index[i], SECOND_COLUMN_OCC],
                                             df_words.at[df_cooccurences.columns[j], SECOND_COLUMN_OCC],
                                             N)


if __name__ == "__main__":
    # Load words with number of occurrences.
    df_words = pd.read_csv(PATH_OCCURRENCES)
    df_words.set_index(FIRST_COLUMN_OCC, inplace=True)

    logs = []

    # Define list of seeds for occurrences (word, seed)
    seeds = []
    with open(PATH_SEEDS_POSITIVE, 'r') as f1, open(PATH_SEEDS_NEGATIVE, 'r') as f2:
        read_lines(f1, seeds)
        read_lines(f2, seeds)

    # Define list of seeds for occurrences (seed, word)
    seeds_inv = ["_" + seed for seed in seeds]

    # Define vector of locks for accessing rows of matrix of occurrences
    locks_vec = []

    # Initialize it with locks
    for _ in range(0, df_words.shape[0]):
        locks_vec.append(threading.Lock())

    # Define dataframe for co-occurrences count
    df_cooccurences = pd.DataFrame(0, index=df_words.index, columns=seeds + seeds_inv)

    # Define thread pool
    executor = ThreadPoolExecutor()

    # Create an iterator to load the dataset in chunks
    data_iterator = pd.read_json(PATH_DATASET, lines=True, chunksize=CHUNK_SIZE)

    futures = [executor.submit(co_occurrences_calc,
                               chunk,
                               locks_vec,
                               df_cooccurences,
                               seeds) for chunk in data_iterator]
    wait(futures)

    # First, compute the total number of occurrences
    N = 0
    for i in range(0, df_words.shape[0]):
        N += df_words.iloc[i, 0]

    # Compute the pmi for every pair row-wise
    futures = [executor.submit(pmi_calc, df_cooccurences, df_words, row_range, N)
               for row_range in interval_generator(0, df_cooccurences.shape[0], CHUNK_SIZE)]
    wait(futures)

    # Now df_cooccurences containes the pmi for each pair. Save dataframe to csv file
    df_cooccurences.to_csv()
