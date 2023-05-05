from nltk.corpus import stopwords
import language_tool_python
import contractions
import re
import spacy
import emoji
from emot.emo_unicode import EMOTICONS_EMO



class TextPreprocessor:
    """
    This class is used to preprocess the text of the tweets.
    It removes the punctuation, stopwords, links and expands the contractions.

    Methods:
        - preprocess_text(text : str) -> str
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tool = language_tool_python.LanguageTool('en-US')
        self.nlp = spacy.load('en_core_web_sm')

    def convert_emojis(self, text:str):
        """
        This function converts emojis to text.

        Args:
            - text : str
        Returns:
            - emoji_str : str
        """
        emoji_str = emoji.demojize(text)
        return emoji_str
    
    def convert_emoticons(self, text:str):
        """
        This function converts emoticons to text.

        Args:
            - text : str
        Returns:
            - emoticons_str : str
        """
        emoticons_str = text
        for word in text.split(" "):
            for emoticon in EMOTICONS_EMO.keys():
                if emoticon in word:
                    emoticons_str = emoticons_str.replace(emoticon, EMOTICONS_EMO[emoticon].split("or")[0].replace(" ", "_").rstrip().lower())

        return emoticons_str
    
    

    def preprocess_text(self, text : str) -> str:
        """
        This function preprocesses the text.
        It removes the punctuation, stopwords, links and expands the contractions.
        Fruthermore, it lemmatizes the text and convert the emojis/emoticons.
        
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

        ## Convert emoticons
        text = self.convert_emoticons(text)

        ## Expand contractions
        text = contractions.fix(text)

        ## Lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_.lower().strip() if "_" not in token.lemma_.lower().strip() else f":{token.lemma_.lower().strip()}:"  for token in doc if not token.is_punct and not token.like_num]

        ## Convert emojis
        tokens = [self.convert_emojis(token) for token in tokens]

        return " ".join(tokens)


## Test:
if __name__ == '__main__':
    text = "I don't want to be a students 😍 :). I'm learnig NLP. @Filippo https://www.google.com  , Sup dude, wanna grab some www.youtube.com grub and chillax at the crib later?"
    text_preprocessor = TextPreprocessor()
    preprocessed_text = text_preprocessor.preprocess_text(text)
    print(preprocessed_text)

    # print(EMOTICONS_EMO[":)"].split("or")[0].replace(" ", "_"))

   
    


    


