import pandas as pd
from Preprocessing.textPreprocessor import TextPreprocessor
from tqdm.auto import tqdm

tqdm.pandas()

class DataFrameManager:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def preprocess_text(self, text):
        return self.preprocessor.preprocess_text(text)
    
    def preprocess_df(self, df):
        decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

        def decode_sentiment(label):
            return decode_map[int(label)]

        df.target = df.target.progress_apply(decode_sentiment)
        df.text = df.text.progress_apply(self.preprocess_text)

        return df


    def load_dataframe(self, filepath:str, encoding=None, names=None, preprocess=True) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding=encoding, names=names)
        if preprocess:
            print("Preprocessing the text...")
            df = self.preprocess_df(df)
        return df

    def export_dataframe(self, df : pd.DataFrame, filepath : str) -> None:
        df.to_csv(filepath, index=False)
