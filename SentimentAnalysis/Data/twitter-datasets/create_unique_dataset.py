import pandas as pd
from DataFrameManager.dataframeManager import DataFrameManager


if __name__ == "__main__":
    data_frame_manager = DataFrameManager(num_cpus=4)

    data_frame_manager.txt_to_csv("train_pos.txt", "train_pos.csv")
    data_frame_manager.txt_to_csv("train_neg.txt", "train_neg.csv")
    pos = pd.read_csv('train_pos_full.csv')
    pos["target"] = "1"
    pos.to_csv('train_pos_full.csv', index=False)
    neg = pd.read_csv('train_neg_full.csv')
    neg["target"] = "-1"
    neg.to_csv('train_neg_full.csv', index=False)
    train_full = pd.concat([pos, neg], axis=0)
    train_full = train_full.drop_duplicates(subset='text')
    train_full.to_csv('train_full.csv', index=False)