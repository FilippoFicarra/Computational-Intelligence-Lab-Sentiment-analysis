# Computational-Intelligence-Lab-Sentiment-analysis
Welcome to the Sentiment analysis project for Computational Intelligence Lab ETH - Spring 2023

First of all you need to setup your enviroment. A virtual enviroment will be created for you with all the dependencies you need to run the project, you just need to run **setup_enviroment.sh**.

# The structure of the project:
```bash
├── Data
│   ├── preprocessed.csv
│   ├── test_embeddings_roberta.npy
│   ├── train_embeddings_roberta.npy
│   ├── training.1600000.processed.noemoticon.csv
│   └── twitter-datasets/
├── DataFrameManager
│   ├── __init__.py
│   └── dataframeManager.py
├── Embedder
│   ├── __init__.py
│   └── embedder.py
├── NGrams
│   ├── __init__.py
│   └── ngrams.py
├── Preprocessing
│   ├── __init__.py
│   └── textPreprocessor.py
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

