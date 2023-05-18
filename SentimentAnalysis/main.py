from DataFrameManager.dataframeManager import DataFrameManager
from Embedder.embedder import Embedder
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"

    PREPROCESSING = False
    CREATE_EMBEDDINGS = True 
    MODEL_NAME = 'roberta'

    data_frame_manager = DataFrameManager(num_cpus=4)
    embed = Embedder(MODEL_NAME)

    if PREPROCESSING:
        if not os.path.exists('SentimentAnalysis/Data/preprocessed.csv'):
            with open('SentimentAnalysis/Data/preprocessed.csv', 'w'):
                pass

        print("Starting preprocessing...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/training.1600000.processed.noemoticon.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        print(df.shape)
        data_frame_manager.export_dataframe(df, filepath="SentimentAnalysis/Data/preprocessed.csv", encoding=DATASET_ENCODING)
        print("Preprocessing done and saved to CSV file.")
        exit()
    else:
        if not os.path.exists('SentimentAnalysis/Data/preprocessed.csv'):
            raise Exception("The file does not exist. Please set PREPROCESSING to True and run the script again.")
        print("Loading the preprocessed data...")
        df = data_frame_manager.load_dataframe(filepath="SentimentAnalysis/Data/preprocessed.csv", encoding=DATASET_ENCODING, preprocess=False)
        print(df.shape)
        print("Splitting the data...")
        train_df, test_df = data_frame_manager.split(df = df)
        print("Data loaded.")
    if CREATE_EMBEDDINGS:
        # Get the embeddings for the test set
        if not os.path.exists(f'SentimentAnalysis/Data/test_embeddings_{MODEL_NAME}.npy'):
            with open(f'SentimentAnalysis/Data/test_embeddings_{MODEL_NAME}.npy', 'wb'):
                pass
        print("Getting the embeddings for the test set...")
        test_embeddings = embed.get_embeddings(test_df['text'])
        print("Saving the embeddings...")

        with open(f'SentimentAnalysis/Data/test_embeddings_{MODEL_NAME}.npy', 'wb') as f:
            np.save(f, test_embeddings.numpy())

        # Get the embeddings for the train set
        if not os.path.exists(f'SentimentAnalysis/Data/train_embeddings_{MODEL_NAME}.npy'):
            with open(f'SentimentAnalysis/Data/train_embeddings_{MODEL_NAME}.npy', 'wb'):
                pass
        print("Getting the embeddings for the train set...")
        train_embeddings = embed.get_embeddings(train_df['text'])
        print("Saving the embeddings...")

        with open(f'SentimentAnalysis/Data/train_embeddings_{MODEL_NAME}.npy', 'wb') as f:
            np.save(f, train_embeddings.numpy())
    else:

        if not os.path.exists(F'SentimentAnalysis/Data/test_embeddings_{MODEL_NAME}.npy') or not os.path.exists(f'SentimentAnalysis/Data/train_embeddings_{MODEL_NAME}.npy'):
            raise Exception("The file does not exist. Please set CREATE_EMBEDDINGS to True and run the script again.")
        
        print("Loading the embeddings...")
        test_embeddings = np.load(f'SentimentAnalysis/Data/test_embeddings_{MODEL_NAME}.npy', allow_pickle=True)
        train_embeddings = np.load(f'SentimentAnalysis/Data/train_embeddings_{MODEL_NAME}.npy', allow_pickle=True)
        print("Embeddings loaded.")


    print("Train embeddings shape: ", train_embeddings.shape)
    print("Test embeddings shape: ", test_embeddings.shape)


"""
# encode_map = {"NEGATIVE" : 0, "NEUTRAL" : 2, "POSITIVE" : 4}
    

    # train_labels = train_df["target"].map(encode_map).to_list()
    # test_labels = test_df["target"].map(encode_map).to_list()


    

    # # Start training
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import GridSearchCV

    # # # Define the hyperparameter grid
    # # param_grid = {
    # #     'C': [0.1, 1.0, 10.0, 100.0],
    # #     'max_iter': [100, 1000, 2500],
    # #     'random_state': [42],

    # # }

    # # # Create the Logistic Regression classifier
    # # classifier_lr = LogisticRegression()

    # # # Perform grid search with cross-validation
    # # grid_search = GridSearchCV(classifier_lr, param_grid, cv=5)
    # # grid_search.fit(train_embeddings, train_labels)

    # # # Get the best hyperparameters and best score
    # # best_params = grid_search.best_params_
    # # best_score = grid_search.best_score_

    # # print("Best Hyperparameters: ", best_params)
    # # print("Best Score: ", best_score)

    # # # Fit the model with the best hyperparameters on the entire training data
    # # best_classifier_lr = LogisticRegression(**best_params)
    # # best_classifier_lr.fit(train_embeddings, train_labels)

    # # # Predict on the test set
    # # predictions_lr = best_classifier_lr.predict(test_embeddings)

    # # # Calculate the accuracy score
    # # accuracy_lr = accuracy_score(test_labels, predictions_lr)
    # # print("Accuracy score for Logistic Regression: ", accuracy_lr)



    # # Start training a Random Forest Classifier
    # from sklearn.ensemble import RandomForestClassifier
    # # Define the hyperparameter grid
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [5, 10],
    #     'min_samples_split': [2, 5, 10]
    # }

    # # Create the Random Forest classifier
    # classifier_rf = RandomForestClassifier()

    # # Perform grid search with cross-validation
    # grid_search = GridSearchCV(classifier_rf, param_grid, cv=5)
    # grid_search.fit(train_embeddings, train_labels)

    # # Get the best hyperparameters and best score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_

    # print("Best Hyperparameters: ", best_params)
    # print("Best Score: ", best_score)

    # # Fit the model with the best hyperparameters on the entire training data
    # best_classifier_rf = RandomForestClassifier(**best_params)
    # best_classifier_rf.fit(train_embeddings, train_labels)

    # # Predict on the test set
    # predictions_rf = best_classifier_rf.predict(test_embeddings)

    # # Calculate the accuracy score
    # accuracy_rf = accuracy_score(test_labels, predictions_rf)
    # print("Accuracy score for Random Forest: ", accuracy_rf)
"""
    
        