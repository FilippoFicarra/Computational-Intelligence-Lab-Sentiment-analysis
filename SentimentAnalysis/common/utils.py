from distutils.errors import PreprocessError

from SentimentAnalysis.Preprocessing.textPreprocessor import TextPreprocessor
from SentimentAnalysis.common  import constants


def wrapper(args: tuple):
    """
    Wrapper to process column of dataframe.
    
    Args:
        - args {tuple} -- the column to process and the function to apply
    
    Returns:
        -  pd.Series -- processed column
    """
    
    df_column, func = args
    return df_column.progress_apply(func)

def preprocess_text(text):
    """
    Wrapper for text preprocessing for multiprocessing.
    
    Args:
        - text {str} -- the string to preprocess
    
    Returns:
        - str: the processed string
    """
    
    return TextPreprocessor(constants.tool, constants.nlp).preprocess_text(text)

def decode_sentiment(label):
    """
    Wrapper for label decoding for multiprocessing.
    
    Args:
        - label {tuple} -- the label in natural language
    
    Returns:
        - int: the label in integer format
    """
    
    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    return decode_map[int(label)]
