# Computational-Intelligence-Lab-Sentiment-analysis
Welcome to the Sentiment analysis project for Computational Intelligence Lab ETH - Spring 2023.

## Data
The data can be downloaded from the following link:

Place datasets and twitter-data in SentimentAnalysis/data, and place model in SentimentAnalysis.

## Data Processing 
Spark is needed to generate the twitter dataset used for training and testing. 
data_processing/twitter-dataset-generator.ipynb is the notebook which is used for generation. All the functions used for
data preprocessing are in the folder data_preprocessing.

## Training 
The script model_train_distributed.py is used to train one of the models of the ensamble on all cores of a TPU. The 
script ensamble_train.py is used to train the ensambler model.

## Evaluation 
The script model_inference.py computes the prediciton on the test dataset for all models with extension .pt placed in 
the folder model. The script ensamble_inference.py produces the predictions for the ensambler model, and it requires 
that three base models and an ensambler of type EnsamblerWithSelfAttention are placed in the folder named model. All 
predictions are placed in the folder named predictions.

## Majority voting 
We explored majority voting. Majority voting can be performed using the majority_voting.py module, which requires that 
all prediction files are placed in the folder named majority.

## Plotting 
Plotting is performed using the script plot.py, which produces the plots for the accuracy and the loss over epochs for 
both training and validation. The files with the accuracies and losses are produced by model_train_distributed.py and 
ensamble_train.py and should be placed in the folder named measures. The results can be found in the folder named 
figures.

## Analysis of predictions 
The analysis of the performance of the models is done using the dataset test_analysis.json in analysis_predictions. 
In order to produce the predictions, it is possible to use model_inference.py and ensamble_inference.py and change 
PATH_DATASET_TWITTER_TEST in CONSTANTS.py to "analysis_predictions/test_analysis.json".

## Baselines
The basline models can be found in the folder named baseline. First run glove.py, then run glove_rnn.py.


_Remark: it is possible to ignore some files which have not been mentioned above. Those files have been used to 
explore different approaches._
