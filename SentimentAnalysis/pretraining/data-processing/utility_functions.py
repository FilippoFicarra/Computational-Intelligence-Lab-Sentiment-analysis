import csv
import os
import re
import shutil
import numpy as np

import contractions
from cleantext import clean
from flashtext import KeywordProcessor
from nltk.tokenize import word_tokenize


# GLOBAL VARIABLES

# Define a dictionary mapping currency symbols to names
CURRENCY_SYMBOLS = {
    '$': ' [USD] ',
    '€': ' [EUR] ',
    '£': ' [GBP] ',
    '¥': ' [JPY] ',
    '₹': ' [INR] ',
    # Add more currency symbols and their names as needed
}

EMOTICONS = {':*': 'kiss', ':-*': 'kiss', ':x': 'kiss', ':-)': 'happy', ':)': 'happy', ':o)': 'happy', ':]': 'happy',
             ':3': 'happy', ':c)': 'happy', ':>': 'happy', '=]': 'happy', '8)': 'happy', '=)': 'happy', ':}': 'happy',
             ':^)': 'happy', '|;-)': 'happy', ":'-)": 'happy', ":')": 'happy', '\\o/': 'happy', '*\\0/*': 'happy',
             ':-D': 'laugh', ':D': 'laugh', '8-D': 'laugh', '8D': 'laugh', 'x-D': 'laugh', 'xD': 'laugh',
             'X-D': 'laugh', 'XD': 'laugh', '=-D': 'laugh', '=D': 'laugh', '=-3': 'laugh', '=3': 'laugh',
             'B^D': 'laugh', '>:[': 'sad', ':-(': 'sad', ':(': 'sad', ':-c': 'sad', ':c': 'sad', ':-<': 'sad',
             ':<': 'sad', ':-[': 'sad', ':[': 'sad', ':{': 'sad', ':-||': 'sad', ':@': 'sad', ":'-(": 'sad',
             ":'(": 'sad', 'D:<': 'sad', 'D:': 'sad', 'D8': 'sad', 'D;': 'sad', 'D=': 'sad', 'DX': 'sad', 'v.v': 'sad',
             "D-':": 'sad', '(>_<)': 'sad', ':|': 'sad', '>:O': 'surprise', ':-O': 'surprise', ':-o': 'surprise',
             ':O': 'surprise', '°o°': 'surprise', 'o_O': 'surprise', 'o_0': 'surprise', 'o.O': 'surprise',
             'o-o': 'surprise', '8-0': 'surprise', '|-O': 'surprise', ';-)': 'wink', ';)': 'wink', '*-)': 'wink',
             '*)': 'wink', ';-]': 'wink', ';]': 'wink', ';D': 'wink', ';^)': 'wink', ':-,': 'wink', '>:P': 'tong',
             ':-P': 'tong', ':P': 'tong', 'X-P': 'tong', 'x-p': 'tong', 'xp': 'tong', 'XP': 'tong', ':-p': 'tong',
             ':p': 'tong', '=p': 'tong', ':-Þ': 'tong', ':Þ': 'tong', ':-b': 'tong', ':b': 'tong', ':-&': 'tong',
             '>:\\': 'annoyed', '>:/': 'annoyed', ':-/': 'annoyed', ':-.': 'annoyed', ':/': 'annoyed', ':\\': 'annoyed',
             '=/': 'annoyed', '=\\': 'annoyed', ':L': 'annoyed', '=L': 'annoyed', ':S': 'annoyed', '>.<': 'annoyed',
             ':-|': 'annoyed', '<:-|': 'annoyed', ':-X': 'seallips', ':X': 'seallips', ':-#': 'seallips',
             ':#': 'seallips', 'O:-)': 'angel', '0:-3': 'angel', '0:3': 'angel', '0:-)': 'angel', '0:)': 'angel',
             '0;^)': 'angel', '>:)': 'devil', '>:D': 'devil', '>:-D': 'devil', '>;)': 'devil', '>:-)': 'devil',
             '}:-)': 'devil', '}:)': 'devil', '3:-)': 'devil', '3:)': 'devil', 'o/\\o': 'highfive', '^5': 'highfive',
             '>_>^': 'highfive', '^<_<': 'highfive', '<3': 'heart', '*:': 'kiss', '*-:': 'kiss', 'x:': 'kiss',
             '(-:': 'happy', '(:': 'happy', '(o:': 'happy', '[:': 'happy', '<:': 'happy', '[=': 'happy', '(=': 'happy',
             '{:': 'happy', "(-':": 'happy', "(':": 'happy', ']:<': 'sad', ')-:': 'sad', '):': 'sad', '>-:': 'sad',
             '>:': 'sad', ']-:': 'sad', ']:': 'sad', '}:': 'sad', '||-:': 'sad', '@:': 'sad', ")-':": 'sad',
             ")':": 'sad', '|:': 'sad', 'O:<': 'surprise', 'O-:': 'surprise', 'o-:': 'surprise', 'O:': 'surprise',
             '.-:': 'annoyed', '|-:': 'annoyed', '|-:>': 'annoyed', '#-:': 'seallips', '#:': 'seallips',
             '(-:O': 'angel', '(-:0': 'angel', '(:0': 'angel', '(:<': 'devil', '(-:<': 'devil', '(-:{': 'devil',
             '(:{': 'devil', ':-d': 'laugh', ':d': 'laugh', '8-d': 'laugh', '8d': 'laugh', 'x-d': 'laugh',
             'xd': 'laugh', '=-d': 'laugh', '=d': 'laugh', 'b^d': 'laugh', 'd:<': 'sad', 'd:': 'sad', 'd8': 'sad',
             'd;': 'sad', 'd=': 'sad', 'dx': 'sad', "d-':": 'sad', '>:o': 'surprise', ':o': 'surprise',
             'o_o': 'surprise', 'o.o': 'surprise', '|-o': 'surprise', ';d': 'wink', '>:p': 'tong', ':-þ': 'tong',
             ':þ': 'tong', ':l': 'annoyed', '=l': 'annoyed', ':s': 'annoyed', ':-x': 'seallips', 'o:-)': 'angel',
             '>:d': 'devil', '>:-d': 'devil', 'o:<': 'surprise', 'o:': 'surprise', '(-:o': 'angel'}

PATH_TO_DOMAINS = "../data/cleaning/domain_extensions.txt"
PATH_TO_FILE_EXTENSIONS = "../data/cleaning/file_extensions.txt"
PATH_TO_PROFANITIES = "../data/cleaning/profanities.txt"
PATH_TO_UNKNOWN = "data/cleaning/unknown_words.txt"

SCHEMA = "overall float, reviewText string"


# FUNCTIONS ON FILE

def read_lines(f, seeds):
    line = f.readline()
    while line != '':
        seeds.append(line.strip("\n"))
        line = f.readline()


def save_list_to_csv(data_list, file_path, attributes) -> None:
    """
    This function saves a list to a csv file.
    Args:
        data_list: the list
        file_path: the path to the csv file
        attributes: the set of attributes of the table

    Returns: None

    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(attributes)  # Optional: Write column headers
        writer.writerows(data_list)


# GLOBAL VARIABLE FOR CLEANING FUNCTIONS

# Get emojis
KEYWORD_PROCESSOR_EMOJIS = KeywordProcessor()
for key, value in EMOTICONS.items():
    KEYWORD_PROCESSOR_EMOJIS.add_keyword(key, value)

# Get domain extensions
domains = []
with open(PATH_TO_DOMAINS, "r") as f:
    read_lines(f, domains)

# Regex subpattern to match urls
SUB_PATTERN_DOMAINS = "|".join(['\\' + domain + '(?=/| )' for domain in domains])

# Get file extensions
file_extensions = []
with open(PATH_TO_FILE_EXTENSIONS, "r") as f:
    read_lines(f, file_extensions)

# Regex subpattern to match urls
SUB_PATTERN_EXTENSIONS = "|".join(['\\' + ext for ext in file_extensions])

profanities = []
with open(PATH_TO_PROFANITIES, "r") as f:
    read_lines(f, profanities)

KEYWORD_PROCESSOR_PROFANITIES = KeywordProcessor()
for profanity in profanities:
    KEYWORD_PROCESSOR_PROFANITIES.add_keyword(profanity, " [BAD] ")


# POLARITY CALCULATOR

def calculate_polarity_of_occurrence(overall) -> int:
    """
    This function calculates the polarity of the occurrence of a word. If the word appears in a negative review, then
    the occurrence receives a negative value, else it receives a positive value. The value are assigned as follows:
    - Overall 1 receives -10.
    - Overall 2 receives -2.
    - Overall 3 receives 0.
    - Overall 4 receives 2.
    - Overall 5 receives 10.
    Args:
        overall: the score of the review.

    Returns: The score assigned to the occurrence.
    """
    base_value = 2

    if overall == 1.:
        return -5 * base_value
    elif overall == 2.:
        return -base_value
    elif overall == 3.:
        return 0
    elif overall == 4.:
        return base_value
    else:
        return 5 * base_value


# FUNCTIONS FOR CLEANING

def replace_xml_tag(string) -> str:
    """
    This function replaces xml tags with the token [XML].
    Args:
        string: The string to modify.

    Returns: The modified string with no xml tags.

    """
    return re.sub(r"<([^>]+)>(?:.*</([^>]+)>)?", " [XML] ", string)


def replace_email(string) -> str:
    """
    This function replaces an email with the token [EMAIL]. It matches also emails which are not completely well-formed.
    Args:
        string: The string to modify.

    Returns: The modified string with no emails.

    """
    match = re.search(r"[a-zA-Z0-9._+-]+\s?@\s?[a-zA-Z0-9.-]+\s?\.\s?com", string)
    if match is not None:
        return re.sub(r"[a-zA-Z0-9._+-]+\s?@\s?[a-zA-Z0-9.-]+\s?\.\s?com", ' [EMAIL] ', string)
    else:
        return string


def replace_link(string) -> str:
    """
    This function replaces links with the token [URL].
    Args:
        string: The string to modify.

    Returns: The modified string with no links.

    """

    pattern = \
        rf"(?:https?://)?(?:www\.)?(?!.*--)[a-zA-Z0-9-]{{1,63}}(?:\.[a-zA-Z0-9-]{{1,63}})*(?:{SUB_PATTERN_DOMAINS})(\
        ?:[^ ]*)?"
    string = re.sub(pattern, ' [URL] ', string)
    pattern = \
        rf"(?:https?://)(?:www\.)?(?!.*--)[a-zA-Z0-9-]{{1,63}}(?:\.[a-zA-Z0-9-]{{1,63}})*(?:{SUB_PATTERN_DOMAINS})?(\
        ?:[^ ]*)?"
    return re.sub(pattern, ' [URL] ', string)


def replace_path(string) -> str:
    """
    This function replaces filenames with the token [URL].
    Args:
        string: The string to modify.

    Returns: The modified string with no links.

    """
    # Regex pattern to match paths
    pattern = rf"(?:/|\\)?(?:[^\s\\/:\*\?\"<>\|.]+[/\\.])*(?:[^\s\\/:\*\?\"<>\|.]+(?:{SUB_PATTERN_EXTENSIONS}))"
    return re.sub(pattern, ' [PATH] ', string)


def replace_emojis(string):
    return KEYWORD_PROCESSOR_EMOJIS.replace_keywords(string)


def replace_special_tokens_with_placeholder(text, placeholder) -> str:
    # Define a regular expression pattern to match sequences of "[text]"
    sequence_pattern = r"\[(?:EMAIL|URL|XML|PATH|NUMBER|USD|EUR|GBP|JPY|INR|BAD|UNKNOWN)\]"

    # Find all matches of the sequence pattern in the text
    sequence_matches = re.findall(sequence_pattern, text)

    # Replace the sequences with a special placeholder
    return re.sub(sequence_pattern, placeholder, text), sequence_matches


def remove_special_characters(string):
    """
    This function replaces special characters with " ". It keeps ".", "!", "?", "%" and ', but if "!", "?", "%" or '
    appear multiple times, then they are replaced with one single occurrence.
    Args:
        string: The string to modify

    Returns: The modified string with no special characters.

    """
    # Note that spaces are managed by clean, * by multiple functions, and currencies by replace_currency_symbols
    placeholder = 'SEQUENCE'
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, placeholder)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s,*.!?%@#$€£¥₹]", " ", text_with_placeholders)
    cleaned_text = re.sub(r"([,!?€£¥₹])\1+", r'\1', cleaned_text)
    cleaned_text = re.sub(r'([.])\1+', '...', cleaned_text)
    if sequence_matches:
        pattern = re.compile(re.escape(placeholder))
        cleaned_text = pattern.sub(lambda _: sequence_matches.pop(0), cleaned_text)
    return cleaned_text


def add_space_before_and_after_punctuation(string):
    # Matches commas, dots, exclamation marks, and question marks
    return re.sub(r" \.\s\s\.\s\s\. ", r" ... ", re.sub(r'([.,!?€£¥₹])', r' \1 ', string))


def replace_profanities(string):
    return KEYWORD_PROCESSOR_PROFANITIES.replace_keywords(string)


def late_remove_special_characters(string):
    return re.sub(r"([*@#])+", "", re.sub(r"([$%])\1+", r' \1 ', string))


def replace_currency_symbols(string):
    """
    This function replaces currency symbols with the corresponding token.
    Args:
        string: The string to modify.

    Returns: The modified string with no currency symbols.

    """
    pattern = re.compile('(%s)' % '|'.join(re.escape(symbol) for symbol in CURRENCY_SYMBOLS.keys()))
    return pattern.sub(lambda x: CURRENCY_SYMBOLS[x.group()], string)


def divide_words_starting_with_numbers(string):
    pattern = r'\b([0-9]+)([a-zA-Z]+)\b'  # regular expression pattern for words starting with a number
    return re.sub(pattern, r'\1 \2', string)


def replace_numbers(string):
    """
    This function replaces numbers with the token [NUMBER].
    Args:
        string: The string to modify.

    Returns: The modified string with no numbers.

    """
    pattern = r'\b\d+(?:-\d+)?\b'  # Regex pattern to match numbers or number ranges
    return re.sub(pattern, ' [NUMBER] ', string)


def cleaning_function_no_unknown(string) -> str:
    return clean(replace_numbers(
        divide_words_starting_with_numbers(
            replace_currency_symbols(
                late_remove_special_characters(
                    replace_profanities(
                        add_space_before_and_after_punctuation(
                            remove_special_characters(
                                replace_emojis(
                                    contractions.fix(
                                        replace_path(
                                            replace_link(
                                                replace_email(
                                                    replace_xml_tag(string.lower()))))))))))))), lower=False,
        no_line_breaks=True)


# WORD TOKENIZER

def remove_symbols_before_tokenization(string):
    # Note that spaces are managed by clean, * by multiple functions, and currencies by replace_currency_symbols
    placeholder = 'SEQUENCE'
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, placeholder)
    cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", text_with_placeholders)
    if sequence_matches:
        pattern = re.compile(re.escape(placeholder))
        cleaned_text = pattern.sub(lambda _: sequence_matches.pop(0), cleaned_text)
    return cleaned_text


def tokenize_with_sequences(string) -> list[str]:
    """
    This function divides a sentence into word tokens.
    Args:
        string: The string to tokenize.

    Returns: A list of tokens.

    """
    placeholder = 'SEQUENCE'
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, placeholder)

    # Tokenize the modified text
    tokens = word_tokenize(text_with_placeholders)

    # Replace the placeholders with the original sequences
    final_tokens = []
    for token in tokens:
        if token == placeholder:
            # Restore the original sequence
            sequence = sequence_matches.pop(0)
            final_tokens.append(sequence)
        else:
            final_tokens.append(token)

    return final_tokens


# FUNCTIONS FOR PMI CALCULATION

def get_pairs_word_seed(tokens, seeds_dict, occurrences_dict, checked_dict):
    pairs = []
    for i in range(0, len(tokens)):
        # Check if current token is a seed
        if seeds_dict[tokens[i]]:
            if i - 1 >= 0 and occurrences_dict[tokens[i - 1]] >= 250 and checked_dict[tokens[i - 1]]:
                pairs.append((tokens[i], tokens[i - 1]))
            if i + 1 < len(tokens) and occurrences_dict[tokens[i + 1]] >= 250 and checked_dict[tokens[i + 1]]:
                pairs.append((tokens[i], tokens[i + 1]))
    # Return the pairs
    return pairs


def pmi(c_w1_w2, c_w1, c_w2, N):
    # Calculate pmi
    result = np.log2((c_w1_w2 * N) / (c_w1 * c_w2))
    if np.isinf(result) or result < 0:
        return 0
    return result


# FUNCTIONS FOR STORING DATASET


def save_rdd_to_json_file(path, rdd):
    # Save cleaned dataset with unknown words
    rdd.toDF(schema=SCHEMA).write.json(path)


def merge_files(path_dataset_directory, path_merged_dataset):
    # Create dataset from json files within path_dataset_directory
    with open(path_merged_dataset, "a+") as f1:
        for root, dirs, files in os.walk(path_dataset_directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    # Read and process file
                    with open(file_path, "r") as f2:
                        f1.writelines(f2.readlines())

    # Remove temporary directory
    shutil.rmtree(path_dataset_directory)
