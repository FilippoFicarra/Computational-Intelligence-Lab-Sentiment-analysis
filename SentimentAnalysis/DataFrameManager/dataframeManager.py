import numpy as np
from common import utils
import multiprocessing
import pandas as pd
# from Preprocessing.textPreprocessor import TextPreprocessor
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split



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
    
    def __init__(self, num_cpus : int = 1):
        self._num_cpus = num_cpus
    
    def preprocess_df(self, df):
        """
        This function maps the labels to the corresponding sentiment and preprocesses the text.

        Args:
            - df : pd.DataFrame
        Returns:
            - df : pd.DataFrame
        """
        
        partitions = np.array_split(df, self._num_cpus)
        
        pool = multiprocessing.Pool(processes=self._num_cpus)
    
        results_target = pool.map(utils.wrapper, [(partition.target, utils.decode_sentiment)  for partition in partitions])
        results_text = pool.map(utils.wrapper, [(partition.text, utils.preprocess_text)  for partition in partitions])
        df.text = pd.concat(results_text)
        df.target = pd.concat(results_target)

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
        
        
        if preprocess:
            df = pd.read_csv(filepath, encoding=encoding, names=names).sample(n=100000, random_state=42).reset_index(drop=True)
            print("Preprocessing the text...")
            df = df.sample(n=df.shape[0], random_state=42)
            df = self.preprocess_df(df)
        else:
            df = pd.read_csv(filepath, encoding=encoding, names=names)
        return df.dropna().reset_index(drop=True)

    def export_dataframe(self, df : pd.DataFrame, filepath : str, encoding=None) -> None:
        """
        This function exports the dataframe to a csv file.

        Args:
            - df : pd.DataFrame
            - filepath : str
        Returns:
            - None
        """
        
        df.to_csv(filepath, index=False, encoding=encoding)
    
    def split(self, df : pd.DataFrame, test_size : float = 0.2, random_state : int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        train_df, test_df = train_test_split(df,  test_size=test_size, random_state=random_state)
    
        return train_df, test_df
