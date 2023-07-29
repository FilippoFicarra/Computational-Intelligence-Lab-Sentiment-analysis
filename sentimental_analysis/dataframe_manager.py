import numpy as np
from common import utils
import multiprocessing
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import csv



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

    def txt_to_csv(self, txt_file, csv_file, delimiter='\n'):
        """
        This function converts a text file to a CSV file.

        Args:
            - txt_file : str
            - csv_file : str
            - delimiter : str
        Returns:
            - None
        """

        # Open the input and output files
        with open(txt_file, 'r') as input_file, open(csv_file, 'w', newline='') as output_file:
            # Create a CSV writer
            csv_writer = csv.writer(output_file)

            # Process each line in the text file
            for line in input_file:
                # Split the line by any delimiters and extract the data
                data = line.strip().split(delimiter)  # Change '\t' to the appropriate delimiter

                # Write the data to the CSV file
                csv_writer.writerow(data)
    
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
    
        if "target" in partitions[0].columns:
            results_target = pool.map(utils.wrapper, [(partition.target, utils.decode_sentiment)  for partition in partitions])
        results_text = pool.map(utils.wrapper, [(partition.text, utils.preprocess_text)  for partition in partitions])
        df.text = pd.concat(results_text)
        if "target" in partitions[0].columns:
            df.target = pd.concat(results_target)
        return df


    def load_dataframe(self, filepath:str, encoding=None, names=None, preprocess=True, test = False) -> pd.DataFrame:
        """
        This function loads the dataframe from a csv file.

        Args:
            - filepath : str
            - encoding : str
            - names : list[str]
            - preprocess : bool
            - test : bool
        Returns:
            - df : pd.DataFrame
        """
        df = pd.read_csv(filepath, encoding=encoding, names=names)

        if preprocess: 
            print("Preprocessing the text...")
            if not test:
                df = df.sample(n=df.shape[0], random_state=42)
            df = self.preprocess_df(df)
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
