import pandas as pd

class DataFrameManager:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_dataframe(self):
        df = pd.read_csv(self.filepath)
        return df
    
    def export_dataframe(self, df, filepath):
        df.to_csv(filepath, index=False)