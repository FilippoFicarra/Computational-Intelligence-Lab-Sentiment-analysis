import sys
from embedder.embedder import Embedder
from .dataframe_manager import DataFrameManager
import os
import numpy as np
import pandas as pd
import click
from termcolor import colored



def preprocess_data(src_filepath : str, dst_filepath : str, test_path : str,  data_frame_manager : DataFrameManager, encoding : str, split_path : str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data and saves it to a CSV file.
    
    Args:
        src_filepath (str): The path to the CSV file containing the data.
        dst_filepath (str): The path to the CSV file where the preprocessed data will be saved.
        data_frame_manager (DataFrameManager): The DataFrameManager object.
        encoding (str): The encoding of the CSV file.

    Returns:
        train_df, test_df (tuple[pd.DataFrame, pd.DataFrame]): The train and test sets.

    """
    
    print("Starting preprocessing...")
    df = data_frame_manager.load_dataframe(filepath=src_filepath, encoding=encoding)
    test_df = data_frame_manager.load_dataframe(filepath=test_path, encoding=encoding, test=True)

    data_frame_manager.export_dataframe(test_df, filepath=split_path+"test_data_preprocessed.csv", encoding=encoding)

    data_frame_manager.export_dataframe(df, filepath=dst_filepath, encoding=encoding)
    print("Preprocessing done and saved to CSV file.")


    print("Splitting the data...")

    train_df, test_df = data_frame_manager.split(df = df)

    data_frame_manager.export_dataframe(train_df, filepath=split_path + "train_preprocessed.csv", encoding=encoding)
    data_frame_manager.export_dataframe(test_df, filepath=split_path + "test_preprocessed.csv", encoding=encoding)

    return train_df, test_df

def load_preprocessed(dst_filepath : str, data_frame_manager : DataFrameManager, encoding : str, split_path : str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the preprocessed data from a CSV file and splits it into train and test sets.

    Args:
        dst_filepath (str): The path to the CSV file containing the preprocessed data.
        data_frame_manager (DataFrameManager): The DataFrameManager object.
        encoding (str): The encoding of the CSV file.
        split_path (str): The path to the folder where the train and test sets will be saved.

    Returns:
        train_df, test_df (tuple[pd.DataFrame, pd.DataFrame]): The train and test sets.

    """

    if os.path.isfile(split_path + "train_preprocessed.csv") and os.path.isfile(split_path + "test_preprocessed.csv"):
        print(colored("The train and test sets already exist. Loading them...", "blue"))
        train_df = data_frame_manager.load_dataframe(filepath=split_path + "train_preprocessed.csv", encoding=encoding, preprocess=False)
        test_df = data_frame_manager.load_dataframe(filepath=split_path + "test_preprocessed.csv", encoding=encoding, preprocess=False)
            
        print(colored("Data loaded.", "blue"))

        return train_df, test_df
    
    if not os.path.isfile(dst_filepath):
        print(colored(f"The file {dst_filepath} does not exist. Please use --preprocess as argument when running the script again.", "red"))
        sys.exit()
        
    print("Loading the preprocessed data...")

    df = data_frame_manager.load_dataframe(filepath=dst_filepath, encoding=encoding, preprocess=False)

    print("Splitting the data...")

    train_df, test_df = data_frame_manager.split(df = df)

    data_frame_manager.export_dataframe(train_df, filepath=split_path + "train_preprocessed.csv", encoding=encoding)
    data_frame_manager.export_dataframe(test_df, filepath=split_path + "test_preprocessed.csv", encoding=encoding)
    

    print(colored("Data loaded.", "blue"))

    return train_df, test_df

def create_embeddings(model_name : str, embedding_path :str, train_df : pd.DataFrame, test_df : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates the embeddings for the train and test sets and saves them to a NPY file.

    Args:
        model_name (str): The name of the model to be used for creating the embeddings.
        embedding_path (str): The path to the folder where the embeddings will be saved.
        train_df (pd.DataFrame): The train set.
        test_df (pd.DataFrame): The test set.

    Returns:
        train_embeddings, test_embeddings (tuple[np.ndarray, np.ndarray]): The train and test embeddings.

    """

    embed = Embedder(model_name)
    # Get the embeddings for the test set
    if not os.path.isfile(f'{embedding_path}test_embeddings_{model_name}.npy'):
        with open(f'{embedding_path}test_embeddings_{model_name}.npy', 'wb'):
            pass
    
    print("Getting the embeddings for the test set...")
    test_embeddings = embed.get_embeddings(test_df['text'], f'{embedding_path}test_embeddings_{model_name}.npy')
    print("Saving the embeddings...")

    # with open(f'{embedding_path}test_embeddings_{model_name}.npy', 'wb') as f:
    #     np.save(f, test_embeddings.numpy())

    # Get the embeddings for the train set
    if not os.path.isfile(f'{embedding_path}train_embeddings_{model_name}.npy'):
        with open(f'{embedding_path}train_embeddings_{model_name}.npy', 'wb'):
            pass
    
    print("Getting the embeddings for the train set...")
    train_embeddings = embed.get_embeddings(train_df['text'], f'{embedding_path}train_embeddings_{model_name}.npy')
    print("Saving the embeddings...")

    # with open(f'{embedding_path}train_embeddings_{model_name}.npy', 'wb') as f:
    #     np.save(f, train_embeddings.numpy())

    return train_embeddings, test_embeddings

def load_embeddings(model_name : str, embedding_path : str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the embeddings from a NPY file.
    
    Args:
        model_name (str): The name of the model used for creating the embeddings.
        embedding_path (str): The path to the folder where the embeddings are saved.
        
    Returns:
        train_embeddings, test_embeddings (tuple[np.ndarray, np.ndarray]): The train and test embeddings.
        
    """
    if not os.path.isfile(f'{embedding_path}test_embeddings_{model_name}.npy') or not os.path.exists(f'{embedding_path}train_embeddings_{model_name}.npy'):
        print(colored("The file does not exist. Please set --embedddings as argument when running the script again.", "red"))
        sys.exit()     
       
    print("Loading the embeddings...")
    test_embeddings = np.load(f'{embedding_path}test_embeddings_{model_name}.npy', allow_pickle=True)
    train_embeddings = np.load(f'{embedding_path}train_embeddings_{model_name}.npy', allow_pickle=True)
    print(colored("Embeddings loaded.", "blue"))

    return train_embeddings, test_embeddings



@click.command()
@click.option('--preprocess', is_flag=True, help='Perform preprocessing')
@click.option('--embeddings', is_flag=True, help='Create embeddings')
@click.option('--model', default='roberta-large', help='Model name')
def main(preprocess : bool, embeddings : bool, model : str) -> None:

    if model not in ['roberta-large', 'bert-large-uncased', 'roberta-base', 'bert-base-uncased']:
        print(colored("The model name is not valid. Please use one of the following: roberta-large, bert-large-uncased, roberta-base, bert-base-uncased.", "red"))
        sys.exit()


    DATASET_ENCODING = None 
    data_frame_manager = DataFrameManager(num_cpus=4)

    src_preprocess_filepath = "data/twitter-datasets/train_full.csv"
    test_filepath = "data/twitter-datasets/test_data.csv"
    dst_preprocess_filepath = "data/twitter-datasets/preprocessed/train_full_preprocessed.csv"
    PATH = "data/twitter-datasets/preprocessed/"

    if preprocess:
        train_df, test_df = preprocess_data(src_preprocess_filepath, dst_preprocess_filepath, test_filepath, data_frame_manager, DATASET_ENCODING, PATH)
    else:
        train_df, test_df = load_preprocessed(dst_preprocess_filepath, data_frame_manager, DATASET_ENCODING, PATH)

    if embeddings:
        create_embeddings(model, PATH, train_df, test_df)


if __name__ == '__main__':
    main()