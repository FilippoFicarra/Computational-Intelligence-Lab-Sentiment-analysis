from DataFrameManager.dataframeManager import DataFrameManager
from Embedder.embedder import Embedder
import os

if '__name__' == '__main__':
    directory_path = 'SentimentAnalysis/Data'
    filename = 'preprocessed.csv'
    file_path = os.path.join(directory_path, filename)
    data_frame_manager = DataFrameManager()

    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    
    if os.path.exists(file_path):
        print("File already exists, loading it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/preprocessed.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS, preprocess=False)
        print("File loaded.")
        print(len(df))
    else:
        print("File does not exist, loading the original dataset and preprocessing it...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/training.1600000.processed.noemoticon.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS).dropna(subset=['text'])
        data_frame_manager.export_dataframe(df, filepath="SentimentAnalysis/Data/preprocessed.csv")
        print("Preprocessing done and saved to CSV file.")
    
    train_df, test_df = data_frame_manager.split(df = df)

    embedder = Embedder()
