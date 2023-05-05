import pandas as pd
from Preprocessing.textPreprocessor import TextPreprocessor
from tqdm.auto import tqdm


tqdm.pandas()


class DataFrameManager:
    """
    This class is used to load the dataframe from a csv file and preprocess the text.
    For preprocessing the text, it uses the TextPreprocessor class.
    
    Methods:
        - preprocess_text(text : str) -> str
        - preprocess_df(df : pd.DataFrame) -> pd.DataFrame
        - load_dataframe(filepath : str, encoding=None, names=None, preprocess=True) -> pd.DataFrame
        - export_dataframe(df : pd.DataFrame, filepath : str) -> None
        - split(df : pd.DataFrame, test_size : float = 0.2) -> (pd.DataFrame, pd.DataFrame)
    """
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def preprocess_text(self, text):
        """
        This function preprocesses the text using the TextPreprocessor class.

        Args:
            - text : str
        Returns:
            - text : str
        """
        return self.preprocessor.preprocess_text(text)
    
    def preprocess_df(self, df):
        """
        This function maps the labels to the corresponding sentiment and preprocesses the text.

        Args:
            - df : pd.DataFrame
        Returns:
            - df : pd.DataFrame
        """
        decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

        def decode_sentiment(label):
            return decode_map[int(label)]

        df.target = df.target.progress_apply(decode_sentiment)
        df.text = df.text.progress_apply(self.preprocess_text)

        return df


    def load_dataframe(self, filepath:str, encoding=None, names=None, preprocess=True) -> pd.DataFrame:
        """
        This function loads the dataframe from a csv file.

        Args:
            - filepath : str
            - encoding : str
            - names : list[str]
            - preprocess : bool
        Returns:
            - df : pd.DataFrame
        """
        df = pd.read_csv(filepath, encoding=encoding, names=names)
        if preprocess:
            print("Preprocessing the text...")
            df = df.sample(n=100000, random_state=42)
            df = self.preprocess_df(df)
        return df

    def export_dataframe(self, df : pd.DataFrame, filepath : str) -> None:
        """
        This function exports the dataframe to a csv file.

        Args:
            - df : pd.DataFrame
            - filepath : str
        Returns:
            - None
        """
        df.to_csv(filepath, index=False)
    
    def split(self, df : pd.DataFrame, train_size : float = 0.8, random_state : int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function splits the dataframe into train and test dataframes.

        Args:
            - df : pd.DataFrame
            - train_size : float
            - random_state : int
        Returns:
            - train_df : pd.DataFrame
            - test_df : pd.DataFrame
        """
        train_df = df.sample(frac=train_size, random_state=random_state)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        return train_df, test_df
