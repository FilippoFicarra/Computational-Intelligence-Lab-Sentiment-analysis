from DataFrameManager.dataframeManager import DataFrameManager
from Embedder.embedder import Embedder
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    DATASET_COLUMNS = ["text", "target"]#["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = None#"ISO-8859-1"

    PREPROCESSING = False
    CREATE_EMBEDDINGS = True 
    TXT_TO_CSV = False
    MODEL_NAME = 'roberta'
    PATH = "SentimentAnalysis/Data/"



    data_frame_manager = DataFrameManager(num_cpus=4)

    if TXT_TO_CSV: # fix this code 
        data_frame_manager.txt_to_csv(PATH+"twitter-datasets/train_pos.txt", PATH+"twitter-datasets/train_pos.csv")
        data_frame_manager.txt_to_csv(PATH+"twitter-datasets/train_neg.txt", PATH+"twitter-datasets/train_neg.csv")
        pos = pd.read_csv('SentimentAnalysis/Data/twitter-datasets/train_pos_full.csv')
        pos["target"] = "1"
        pos.to_csv('SentimentAnalysis/Data/twitter-datasets/train_pos_full.csv', index=False)
        neg = pd.read_csv('SentimentAnalysis/Data/twitter-datasets/train_neg_full.csv')
        neg["target"] = "-1"
        neg.to_csv('SentimentAnalysis/Data/twitter-datasets/train_neg_full.csv', index=False)
        train_full = pd.concat([pos, neg], axis=0)
        train_full = train_full.drop_duplicates(subset='text')
        train_full.to_csv('SentimentAnalysis/Data/twitter-datasets/train_full.csv', index=False)

    if PREPROCESSING:
        filepath = PATH+"twitter-datasets/train_full.csv"
        print("Starting preprocessing...")
        df = data_frame_manager.load_dataframe(filepath=filepath, encoding=DATASET_ENCODING)
        print(df.shape)
        data_frame_manager.export_dataframe(df, filepath=PATH+"twitter-datasets/preprocessed/train_full_preprocessed.csv", encoding=DATASET_ENCODING)
        print("Preprocessing done and saved to CSV file.")
        exit()
    else:
        if not os.path.exists('SentimentAnalysis/Data/twitter-datasets/preprocessed/train_full_preprocessed.csv'):
            raise Exception("The file does not exist. Please set PREPROCESSING to True and run the script again.")
        print("Loading the preprocessed data...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/twitter-datasets/preprocessed/train_full_preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
        print(df.shape)
        print("Splitting the data...")
        train_df, test_df = data_frame_manager.split(df = df)
        # train_df = data_frame_manager.export_dataframe(train_df, filepath="SentimentAnalysis/Data/twitter-datasets/preprocessed/train_preprocessed.csv", encoding=DATASET_ENCODING)
        # test_df = data_frame_manager.export_dataframe(test_df, filepath="SentimentAnalysis/Data/twitter-datasets/preprocessed/test_preprocessed.csv", encoding=DATASET_ENCODING)
        print("Data loaded.")
    if CREATE_EMBEDDINGS:
        embed = Embedder(MODEL_NAME)

        # Get the embeddings for the test set
        if not os.path.exists(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/test_embeddings_{MODEL_NAME}.npy'):
            with open(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/test_embeddings_{MODEL_NAME}.npy', 'wb'):
                pass
        print("Getting the embeddings for the test set...")
        test_embeddings = embed.get_embeddings(test_df['text'])
        print("Saving the embeddings...")

        with open(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/test_embeddings_{MODEL_NAME}.npy', 'wb') as f:
            np.save(f, test_embeddings.numpy())

        # Get the embeddings for the train set
        if not os.path.exists(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/train_embeddings_{MODEL_NAME}.npy'):
            with open(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/train_embeddings_{MODEL_NAME}.npy', 'wb'):
                pass
        print("Getting the embeddings for the train set...")
        train_embeddings = embed.get_embeddings(train_df['text'])
        print("Saving the embeddings...")

        with open(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/train_embeddings_{MODEL_NAME}.npy', 'wb') as f:
            np.save(f, train_embeddings.numpy())
    else:

        if not os.path.exists(F'SentimentAnalysis/Data/twitter-datasets/preprocessed/test_embeddings_{MODEL_NAME}.npy') or not os.path.exists(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/train_embeddings_{MODEL_NAME}.npy'):
            raise Exception("The file does not exist. Please set CREATE_EMBEDDINGS to True and run the script again.")
        
        print("Loading the embeddings...")
        test_embeddings = np.load(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/test_embeddings_{MODEL_NAME}.npy', allow_pickle=True)
        train_embeddings = np.load(f'SentimentAnalysis/Data/twitter-datasets/preprocessed/train_embeddings_{MODEL_NAME}.npy', allow_pickle=True)
        print("Embeddings loaded.")


    # print("Train embeddings shape: ", train_embeddings.shape)
    # print("Test embeddings shape: ", test_embeddings.shape)

    # print(df.head(5))

   

