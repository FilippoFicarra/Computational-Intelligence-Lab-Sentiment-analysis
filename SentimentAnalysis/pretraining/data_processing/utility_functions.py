import csv
import os
import re
import shutil

import contractions
import numpy as np
import wordninja
from cleantext import clean
from flashtext import KeywordProcessor
from nltk.tokenize import word_tokenize

# GLOBAL VARIABLES

# Define a dictionary mapping currency symbols to names
CURRENCY_SYMBOLS = {
    '$': ' [CUR] ',
    '€': ' [CUR] ',
    '£': ' [CUR] ',
    '¥': ' [CUR] ',
    '₹': ' [CUR] ',
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
             ':-P': 'tong', ':P': 'tong', 'X-P': 'tong', 'x-p': 'tong', 'xp': 'tong', 'xpp': 'tong', 'XP': 'tong',
             ':-p': 'tong', ':p': 'tong', '=p': 'tong', ':-Þ': 'tong', ':Þ': 'tong', ':-b': 'tong', ':b': 'tong',
             ':-&': 'tong', '>:\\': 'annoyed', '>:/': 'annoyed', ':-/': 'annoyed', ':-.': 'annoyed', ':/': 'annoyed',
             ':\\': 'annoyed', '=/': 'annoyed', '=\\': 'annoyed', ':L': 'annoyed', '=L': 'annoyed', ':S': 'annoyed',
             '>.<': 'annoyed', ':-|': 'annoyed', '<:-|': 'annoyed', ':-X': 'seallips', ':X': 'seallips',
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
             '>:d': 'devil', '>:-d': 'devil', 'o:<': 'surprise', 'o:': 'surprise', '(-:o': 'angel', ':-#': 'seallips'}

ABBREVIATIONS = {
    "4ao": "for adults only",
    "a.m": "before midday",
    "a3": "anytime anywhere anyplace",
    "aamof": "as a matter of fact",
    "acct": "account",
    "adih": "another day in hell",
    "afaic": "as far as i am concerned",
    "afaict": "as far as i can tell",
    "afaik": "as far as i know",
    "afair": "as far as i remember",
    "afk": "away from keyboard",
    "app": "application",
    "approx": "approximately",
    "apps": "applications",
    "asap": "as soon as possible",
    "asl": "age, sex, location",
    "atk": "at the keyboard",
    "ave.": "avenue",
    "aymm": "are you my mother",
    "ayor": "at your own risk",
    "b&b": "bed and breakfast",
    "b+b": "bed and breakfast",
    "b.c": "before christ",
    "b2b": "business to business",
    "b2c": "business to customer",
    "b4": "before",
    "b4n": "bye for now",
    "b@u": "back at you",
    "bae": "before anyone else",
    "bak": "back at keyboard",
    "bbbg": "bye bye be good",
    "bbias": "be back in a second",
    "bbl": "be back later",
    "bbs": "be back soon",
    "be4": "before",
    "bfn": "bye for now",
    "blvd": "boulevard",
    "bout": "about",
    "brb": "be right back",
    "bros": "brothers",
    "brt": "be right there",
    "bsaaw": "big smile and a wink",
    "btw": "by the way",
    "bwl": "bursting with laughter",
    "c/o": "care of",
    "cet": "central european time",
    "cf": "compare",
    "cia": "central intelligence agency",
    "csl": "can not stop laughing",
    "cu": "see you",
    "cul8r": "see you later",
    "cv": "curriculum vitae",
    "cwot": "complete waste of time",
    "cya": "see you",
    "cyt": "see you tomorrow",
    "dae": "does anyone else",
    "dbmib": "do not bother me i am busy",
    "diy": "do it yourself",
    "dm": "direct message",
    "dwh": "during work hours",
    "e123": "easy as one two three",
    "eet": "eastern european time",
    "eg": "example",
    "embm": "early morning business meeting",
    "encl": "enclosed",
    "encl.": "enclosed",
    "etc": "and so on",
    "faq": "frequently asked questions",
    "fawc": "for anyone who cares",
    "fb": "facebook",
    "fc": "fingers crossed",
    "fig": "figure",
    "fimh": "forever in my heart",
    "ft.": "feet",
    "ft": "featuring",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "g9": "genius",
    "gahoy": "get a hold of yourself",
    "gal": "get a life",
    "gcse": "general certificate of secondary education",
    "gfn": "gone for now",
    "gg": "good game",
    "gl": "good luck",
    "glhf": "good luck have fun",
    "gmt": "greenwich mean time",
    "gmta": "great minds think alike",
    "gn": "good night",
    "g.o.a.t": "greatest of all time",
    "goat": "greatest of all time",
    "goi": "get over it",
    "gps": "global positioning system",
    "gr8": "great",
    "gratz": "congratulations",
    "gyal": "girl",
    "h&c": "hot and cold",
    "hp": "horsepower",
    "hr": "hour",
    "hrh": "his royal highness",
    "ht": "height",
    "ibrb": "i will be right back",
    "ic": "i see",
    "icq": "i seek you",
    "icymi": "in case you missed it",
    "idc": "i do not care",
    "idgadf": "i do not give a damn fuck",
    "idgaf": "i do not give a fuck",
    "idk": "i do not know",
    "ie": "that is",
    "i.e": "that is",
    "ifyp": "i feel your pain",
    "IG": "instagram",
    "iirc": "if i remember correctly",
    "ilu": "i love you",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "imu": "i miss you",
    "iow": "in other words",
    "irl": "in real life",
    "j4f": "just for fun",
    "jic": "just in case",
    "jk": "just kidding",
    "jsyk": "just so you know",
    "l8r": "later",
    "lb": "pound",
    "lbs": "pounds",
    "ldr": "long distance relationship",
    "lmao": "laugh my ass off",
    "lmfao": "laugh my fucking ass off",
    "lol": "laughing out loud",
    "ltd": "limited",
    "ltns": "long time no see",
    "m8": "mate",
    "mf": "motherfucker",
    "mfs": "motherfuckers",
    "mfw": "my face when",
    "mofo": "motherfucker",
    "mph": "miles per hour",
    "mr": "mister",
    "mrw": "my reaction when",
    "ms": "miss",
    "mte": "my thoughts exactly",
    "nagi": "not a good idea",
    "nbc": "national broadcasting company",
    "nbd": "not big deal",
    "nfs": "not for sale",
    "ngl": "not going to lie",
    "nhs": "national health service",
    "nrn": "no reply necessary",
    "nsfl": "not safe for life",
    "nsfw": "not safe for work",
    "nth": "nice to have",
    "nvr": "never",
    "nyc": "new york city",
    "oc": "original content",
    "og": "original",
    "ohp": "overhead projector",
    "oic": "oh i see",
    "omdb": "over my dead body",
    "omg": "oh my god",
    "omw": "on my way",
    "p.a": "per annum",
    "p.m": "after midday",
    "poc": "people of color",
    "pov": "point of view",
    "pp": "pages",
    "ppl": "people",
    "prw": "parents are watching",
    "ps": "postscript",
    "pt": "point",
    "ptb": "please text back",
    "pto": "please turn over",
    "qpsa": "what happens",  # "que pasa",
    "ratchet": "rude",
    "rbtl": "read between the lines",
    "rlrt": "real life retweet",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off",
    "rt": "retweet",
    "ruok": "are you ok",
    "sfw": "safe for work",
    "sk8": "skate",
    "smh": "shake my head",
    "sq": "square",
    "srsly": "seriously",
    "ssdd": "same stuff different day",
    "tbh": "to be honest",
    "tbs": "tablespooful",
    "tbsp": "tablespooful",
    "tfw": "that feeling when",
    "thks": "thank you",
    "tho": "though",
    "thx": "thank you",
    "tia": "thanks in advance",
    "til": "today i learned",
    "tl;dr": "too long i did not read",
    "tldr": "too long i did not read",
    "tmb": "tweet me back",
    "tntl": "trying not to laugh",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "u4e": "yours for ever",
    "utc": "coordinated universal time",
    "w/": "with",
    "w/o": "without",
    "w8": "wait",
    "wassup": "what is up",
    "wb": "welcome back",
    "wtf": "what the fuck",
    "wtg": "way to go",
    "wtpa": "where the party at",
    "wuf": "where are you from",
    "wuzup": "what is up",
    "wywh": "wish you were here",
    "yd": "yard",
    "ygtr": "you got that right",
    "ynk": "you never know",
    "zzz": "sleeping bored and tired"
}

STOPWORDS = []

PATH_TO_DOMAINS = "../data/cleaning/domain_extensions.txt"
PATH_TO_FILE_EXTENSIONS = "../data/cleaning/file_extensions.txt"
PATH_TO_PROFANITIES = "../data/cleaning/profanities.txt"
PATH_TO_UNKNOWN = "../data/cleaning/unknown_words.txt"

SCHEMA = "label int, text string"

PLACEHOLDER = 'SEQUENCE'


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
SUB_PATTERN_DOMAINS = "|".join(['\\' + domain for domain in domains])

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

KEYWORD_PROCESSOR_ABBREVIATIONS = KeywordProcessor()
for abbreviation, value in ABBREVIATIONS.items():
    KEYWORD_PROCESSOR_ABBREVIATIONS.add_keyword(abbreviation, value)

KEYWORD_PROCESSOR_STOPWORDS = KeywordProcessor()
for stopword in STOPWORDS:
    KEYWORD_PROCESSOR_STOPWORDS.add_keyword(stopword, " ")


# POLARITY CALCULATOR

def calculate_polarity_of_occurrence(label) -> int:
    """
    This function calculates the polarity of the occurrence of a word. If the word appears in a negative review, then
    the occurrence receives a negative value, else it receives a positive value. The value are assigned as follows:
    - Label 0 receives -1.
    - Label 1 receives 1.
    Args:
        label: the score of the review.

    Returns: The score assigned to the occurrence.
    """

    if label == 0:
        return -1
    else:
        return 1


# FUNCTIONS FOR CLEANING

def replace_xml_tag(string) -> str:
    """
    This function replaces xml tags with the token [XML].
    Args:
        string: The string to modify.

    Returns: The modified string with no xml tags.

    """
    return re.sub(r"<([^>]+)>(?:.*</([^>]+)>)?", ' [XML] ', string)


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
        ?:/[^ ]*)?"
    string = re.sub(pattern, ' [URL] ', string)
    pattern = \
        rf"(?:https?://)(?:www\.)?(?!.*--)[a-zA-Z0-9-]{{1,63}}(?:\.[a-zA-Z0-9-]{{1,63}})*(?:{SUB_PATTERN_DOMAINS})?(\
        ?:/[^ ]*)?"
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


def replace_abbreviations(string):
    return KEYWORD_PROCESSOR_ABBREVIATIONS.replace_keywords(string)


def replace_special_tokens_with_placeholder(string, twitter):
    # Choose pattern
    if not twitter:
        sequence_pattern = r"\[(?:EMAIL|URL|XML|PATH|NUMBER|CUR|BAD)\]"
    else:
        sequence_pattern = r"\[(?:EMAIL|URL|XML|PATH|NUMBER|CUR|BAD)\]|<(?:url|user)>"

    # Find all matches of the sequence pattern in the text
    sequence_matches = re.findall(sequence_pattern, string)

    # Replace the sequences with a special PLACEHOLDER
    return re.sub(sequence_pattern, PLACEHOLDER, string), sequence_matches


def remove_special_characters(string, twitter=False):
    """
    This function replaces special characters with " ". It keeps ".", "!", "?", "%" and ', but if "!", "?", "%" or '
    appear multiple times, then they are replaced with one single occurrence.
    Args:
        string: The string to modify
        twitter: A boolean

    Returns: The modified string with no special characters.

    """
    # Note that spaces are managed by clean, * by multiple functions, and currencies by replace_currency_symbols
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, twitter)

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s,*.!?%@#$€£¥₹]", " ", text_with_placeholders)
    cleaned_text = re.sub(r"([,!?€£¥₹])\1+", r'\1', cleaned_text)
    cleaned_text = re.sub(r'([.])\1+', '...', cleaned_text)
    if sequence_matches:
        pattern = re.compile(re.escape(PLACEHOLDER))
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


def divide_words_starting_with_numbers(string, twitter=False):
    pattern = r'\b([0-9]+)([a-zA-Z]+)\b'  # regular expression pattern for words starting with a number
    if not twitter:
        return re.sub(pattern, r'[NUMBER] \2', string)
    else:
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


def remove_stopwords(string):
    return KEYWORD_PROCESSOR_STOPWORDS.replace_keywords(string)


def cleaning_function_no_unknown(string) -> str:
    return clean(
        replace_numbers(
            divide_words_starting_with_numbers(
                replace_currency_symbols(
                    late_remove_special_characters(
                        replace_profanities(
                            add_space_before_and_after_punctuation(
                                remove_special_characters(
                                    replace_abbreviations(
                                        replace_emojis(
                                            contractions.fix(
                                                replace_path(
                                                    replace_link(
                                                        replace_email(
                                                            replace_xml_tag(string.lower())))))))))))))),
        lower=False,
        no_line_breaks=True)


# FUNCTION FOR CLEANING OF TWITTER DATASET


def late_remove_special_characters_twitter(string):
    return re.sub(r"([*@])+", "", re.sub(r"([$%])\1+", r' \1 ', re.sub(r"(#)\1+", r'\1', string)))


def replace_hashtags(string):
    # Use the regex pattern to find words starting with "#"
    pattern = r'\B#\w+\b'
    hashtags = re.findall(pattern, string)
    kp = KeywordProcessor()
    kp.add_keyword("howamigonnagotocollege", "how am i going to go to college")

    for hashtag in hashtags:
        if hashtag != "howamigonnagotocollege":
            kp.add_keyword(hashtag, " ".join(wordninja.split(hashtag)))

    return kp.replace_keywords(string)


def remove_hashtags(string):
    return re.sub(r"(#)", "", string)


def remove_numbers(string):
    pattern = r'\b\d+(?:-\d+)?\b'  # Regex pattern to match numbers or number ranges
    return re.sub(pattern, '', string)


def cleaning_function_twitter_dataset(string) -> str:
    return divide_words_starting_with_numbers(
        remove_numbers(
                remove_hashtags(
                    replace_hashtags(
                        clean(
                            replace_currency_symbols(
                                late_remove_special_characters_twitter(
                                    replace_profanities(
                                        add_space_before_and_after_punctuation(
                                            remove_special_characters(
                                                replace_abbreviations(
                                                    replace_emojis(
                                                        contractions.fix(
                                                            string.lower()))), twitter=True))))), lower=False,
                            no_line_breaks=True)))), True)


def new_cleaning_function_twitter_dataset(string) -> str:
    return divide_words_starting_with_numbers(
        remove_hashtags(
            replace_hashtags(
                clean(
                    late_remove_special_characters_twitter(
                        add_space_before_and_after_punctuation(
                            remove_special_characters(
                                replace_abbreviations(
                                    replace_emojis(
                                        contractions.fix(
                                            string.lower()))), twitter=True))),
                    lower=False,
                    no_line_breaks=True))),
        True)


# WORD TOKENIZER

def remove_symbols_before_tokenization(string, twitter=False):
    # Note that spaces are managed by clean, * by multiple functions, and currencies by replace_currency_symbols
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, twitter)
    cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", text_with_placeholders)
    if sequence_matches:
        pattern = re.compile(re.escape(PLACEHOLDER))
        cleaned_text = pattern.sub(lambda _: sequence_matches.pop(0), cleaned_text)
    return cleaned_text


def tokenize_with_sequences(string, twitter=False) -> list[str]:
    """
    This function divides a sentence into word tokens.
    Args:
        string: The string to tokenize.
        twitter: A boolean

    Returns: A list of tokens.

    """
    text_with_placeholders, sequence_matches = replace_special_tokens_with_placeholder(string, twitter)

    # Tokenize the modified text
    tokens = word_tokenize(text_with_placeholders)

    # Replace the placeholders with the original sequences
    final_tokens = []
    for token in tokens:
        if token == PLACEHOLDER:
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
        try:
            if seeds_dict[tokens[i]]:
                if i - 1 >= 0 and occurrences_dict[tokens[i - 1]] >= 250 and checked_dict[tokens[i - 1]]:
                    pairs.append((tokens[i], tokens[i - 1]))
                if i + 1 < len(tokens) and occurrences_dict[tokens[i + 1]] >= 250 and checked_dict[tokens[i + 1]]:
                    pairs.append((tokens[i], tokens[i + 1]))
        except Exception:
            pass

    # Return the pairs
    return pairs


def pmi(c_w1_w2, c_w1, c_w2, N):
    # Calculate pmi
    result = np.log2((c_w1_w2 * N) / (c_w1 * c_w2))
    if np.isinf(result) or result < 0:
        return 0
    return result


# FUNCTIONS FOR STORING DATASET


def save_rdd_to_json_file(path, rdd, schema=SCHEMA):
    # Save cleaned dataset with unknown words
    rdd.toDF(schema=schema).write.json(path)


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
