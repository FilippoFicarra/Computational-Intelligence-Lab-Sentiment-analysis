from nltk.corpus import stopwords
import language_tool_python
import contractions
import string
import re

class TextPreprocessor:
    """
    This class is used to preprocess the text of the tweets.
    It corrects the spelling, removes the punctuation, stopwords, links and expands the contractions.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
       
        self.punctuation = set(string.punctuation) 
        self.punctuation.remove("'")
        self.punctuation.remove('"')
        self.punctuation.remove('?')
        self.punctuation.remove('!')

        self.tool = language_tool_python.LanguageTool('en-US')

    def preprocess_text(self, text : str) -> str:

        ## Remove links
        # TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S"
        # text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

        ## Correct spelling
        text = self.tool.correct(text).lower()       

        ## Remove punctuation
        text = ''.join([c for c in text if c not in self.punctuation])

        ## Remove stopwords
        text = ' '.join([w for w in text.split() if w not in self.stop_words])

        ## Expand contractions
        text = contractions.fix(text)

        # It could be useful to lemmatize and use part of speech tagging in order to better compare and translate the words in the vector space

        return text


if __name__ == '__main__':
    text = "I'm a students. I'm learnig NLP."
    text = "I'm a students. I'm learnig NLP. https://www.google.com"
    text_preprocessor = TextPreprocessor()
    preprocessed_text = text_preprocessor.preprocess_text(text)
    print(preprocessed_text)

    


