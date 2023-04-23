# Computational-Intelligence-Lab-Sentiment-analysis
Welcome to the Sentiment analysis project for Computational Intelligence Lab ETH - Spring 2023

First of all you need to setup your enviroment. A virtual enviroment will be created for you with all the dependencies you need to run the project, you just need to run **setup_enviroment.sh**.

# The structure of the project:
```bash
SentimentAnalysis
├── Data
│   ├── preprocessed.csv
│   ├── embeddings.npy
│   └── training.1600000.processed.noemoticon.csv
├── DataFrameManager
│   └── dataframeManager.py
├── Embedder
│   └── embedder.py
├── NGrams
│   └── ngrams.py
└── Preprocessing
    └── textPreprocessor.py

```

- Data: It contains all the data used in the process
- DataFrameManager: It contains a manager to load and create dataframes from the file in the folder Data
- Embedder: It contains a word embedder
- NGrams: It contains a ngram creator from embeddings
- Preprocessing: It contains all the preprocessing functions

