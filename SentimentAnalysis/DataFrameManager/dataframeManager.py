import pandas as pd


class DataFrameManager:
    def __init__(self):
        pass

    def load_dataframe(self, filepath:str, encoding=None, names= None | list) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding=encoding, names=names)
        decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
        def decode_sentiment(label):
            return decode_map[int(label)]
        df.target = df.target.apply(lambda x: decode_sentiment(x))
        return df
    
    def export_dataframe(self, df : pd.DataFrame, filepath : str) -> None:
        df.to_csv(filepath, index=False)