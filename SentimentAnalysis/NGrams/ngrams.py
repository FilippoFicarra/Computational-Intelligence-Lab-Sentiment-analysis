import numpy as np
import torch
class NGrams:
    """
    This class is used to generate n-grams from a sequence of word embeddings.
    
    Methods:
        - generate_ngrams(embeddings : list, n : int) -> list
    """
    def __init__(self):
        pass

    def generate_ngrams(self, embeddings : list, n : int) -> list:
        """
        Generates n-grams from a sequence of word embeddings.

        Args:
            - embeddings: a list or array of word embeddings
            - n: the length of the n-grams

        Returns:
            - ngrams: a list of n-grams
        """
        ngrams = []
        for i in range(len(embeddings)-n+1):
            ngram = np.concatenate([embeddings[j] for j in range(i,i+n)])
            ngrams.append(ngram)
        return torch.tensor(ngrams)