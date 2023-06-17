# Computational-Intelligence-Lab-Sentiment-analysis
Welcome to the Sentiment analysis project for Computational Intelligence Lab ETH - Spring 2023

First of all you need to setup your enviroment. A virtual enviroment will be created for you with all the dependencies you need to run the project, you just need to run **setup_enviroment.sh**.

# The structure of the project:
```bash
├── data
│   ├── preprocessed.csv
│   ├── test_embeddings_roberta.npy
│   ├── train_embeddings_roberta.npy
│   └── twitter-datasets/
├── data_frame_manager
│   ├── __init__.py
│   └── data_frame_manager.py
├── embedder
│   ├── __init__.py
│   └── embedder.py
├── ngrams
│   ├── __init__.py
│   └── ngrams.py
├── preprocessor
│   ├── __init__.py
│   └── text_preprocessor.py
├── __init__.py
├── common
│   ├── constants.py
│   └── utils.py
├── config.py
├── ethz-cil-text-classification-2023/
├── main.py
└── train.ipynb
```

- Data: It contains all the data used in the process
- DataFrameManager: It contains a manager to load and create dataframes from the file in the folder Data
- Embedder: It contains a word embedder (using either BERT or RoBERTa model)
- NGrams: It contains a ngram creator from embeddings
- Preprocessing: It contains all the preprocessing methods
- main.py: It preprocesses the data and create the embeddings
- train.ipynb: It contains the model to train

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

