from nltk.corpus import stopwords
import language_tool_python
import contractions
import re
import spacy

class TextPreprocessor:
    """
    This class is used to preprocess the text of the tweets.
    It corrects the spelling, removes the punctuation, stopwords, links and expands the contractions.
    Functions:
        - preprocess_text(text : str) -> str
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tool = language_tool_python.LanguageTool('en-US')
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_text(self, text : str) -> str:
        """
        This function preprocesses the text.
        Args:
            - text : str
        Returns:
            - text : str
        """
        ## lowercase
        text = text.lower()

        ## Remove links
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S+|www\.\S+"
        text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

        ## Correct spelling
        text = self.tool.correct(text)

        ## Expand contractions
        text = contractions.fix(text)

        ## Lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_.lower().strip() for token in doc if not token.is_punct and not token.like_num]

        return " ".join(tokens)


## Test:
if __name__ == '__main__':
    text = "I don't want to be a students. I'm learnig NLP. https://www.google.com  www.youtube.com, Sup dude, wanna grab some grub and chillax at the crib later?"
    text_preprocessor = TextPreprocessor()
    preprocessed_text = text_preprocessor.preprocess_text(text)
    print(preprocessed_text)
    

   
    


    


