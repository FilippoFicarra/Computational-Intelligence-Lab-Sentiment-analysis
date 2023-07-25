# Computational-Intelligence-Lab-Sentiment-analysis
Welcome to the Sentiment analysis project for Computational Intelligence Lab ETH - Spring 2023

First of all you need to setup your enviroment. A virtual enviroment will be created for you with all the dependencies you need to run the project, you just need to run **setup_enviroment.sh**.

# The structure of the project:
```bash
├── README.md
├── ReferencePapers
├── SentimentAnalysis
│   ├── CONSTANTS.py
│   ├── average_meter.py
│   ├── bert_tweet_sparsemax.py
│   ├── bert_tweet_with_mask.py
│   ├── data
│   │   ├── cleaning
│   │   │   ├── domain_extensions.txt
│   │   │   └── file_extensions.txt
│   │   └── sentiment-knowledge
│   │       ├── seeds-negative.txt
│   │       └── seeds-positive.txt
│   ├── data_processing
│   │   ├── data-cleaning.ipynb
│   │   ├── data-processing.ipynb
│   │   ├── dataset-two-classes-generator.ipynb
│   │   ├── seeds-finder.ipynb
│   │   ├── sentiment-knowledge-miner.ipynb
│   │   ├── twitter-dataset-generator.ipynb
│   │   └── utility_functions.py
│   ├── dataset.py
│   ├── masker.py
│   ├── model_train_distributed.py
│   └── tokenizer_train.py
└── setup_enviroment.sh
```

- data_preprocessing: It contains all the preprocessing applied to the dataset
- average_meter.py: It constains all the metrics used for the distributed computing
- bert_tweet_sparsemax.py : It contains Bertweet model with a modified RobertaSelfAttention using Sparsemax instead of softmax
- model_train_distributed.py : It performs the training in a TPU distributed enviroment on Google Cloud
- bert_tweet_with_mask.py: It contains Bertweet model with a custom masked technique
- dataset.py: It contains the dataset for the twitter datasets and for the masking training with amazon reviews 

# Data
The data given to us are located in Data/twitter-datasets. They consist of:
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

In order to create a unique data set run:

```bash
python3 Data/twitter-datasets/create_unique_dataset.py
```

This will merge all the positive and negative sentences into a unique dataframe. It also adds the column target that is "1" for the positive sentences and "-1" for the negative ones.

NOTE: some sentences strangely are labeled both as negative and positive. They are just discarded when running the script.

# Start the project
Start the virtual enviroment.

Add the SentimentAnalysis as working directory.

```bash
export PYTHONPATH="${PYTHONPATH}:SentimentAnalysis" 
```

Move to SentimentAnalysis

```bash
cd SentimentAnalysis 
```

Run the python script

```bash
python3 main.py
```
If no arguments are passed the main will load the pre-computed traindf, test df and the corrispondent RoBERTa embeddings.

The arguments are the following:

- --preprocess : to preprocess the data
- --embeddings : to create the embeddings
- --model model_name : to choose the model for the embedder (RoBERTa or BERT are the embedder supported)

# Embedder
The embedder get the CLS embedding of the last layer for the model choosen. The available models for now are:

- bert-base-uncased
- roberta-base
- bert-large-uncased
- roberta-large

