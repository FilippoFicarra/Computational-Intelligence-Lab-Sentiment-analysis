import re
# from nltk.corpus import stopwords
# from common import constants
import contractions
import emoji
from emot.emo_unicode import EMOTICONS_EMO

EMOTICONS = EMOTICONS_EMO
 
emoticons = {
        "=|": "emotneutral",
        "=-(": "emotfrown",
        "=-)": "emotsmile",
        "=:": "emotneutral",
        "=/": "emotfrown",
        "='(": "emotfrown",
        "='[": "emotfrown",
        "=(": "emotfrown",
        "=)": "emotsmile",
        "=[": "emotfrown",
        "=]": "xemotsmile",
        "={": "emotfrown",
        "=\\": "emotfrown",
        ">=(": "emotfrown",
        ">=)": "emotsmile",
        ">:|": "emotneutral",
        ">:/": "xxemotfrown",
        ">:[": "emotfrown",
        ">:": "emotfrown",
        "|:": "emotneutral",
        ";|": "emotneutral",
        ";-}": "emotsmile",
        ";:": "emotneutral",
        ";/": "emotfrown",
        ";'/": "emotfrown",
        ";'(": "emotfrown",
        ";')": "emotsmile",
        ";)": "emotsmile",
        ";]": "emotsmile",
        ";}": "emotsmile",
        ";*{": "emotfrown",
        ":|": "emotneutral",
        ":-|": "emotneutral",
        ":-/": "emotfrown",
        ":-[": "emotfrown",
        ":-]": "emotsmile",
        ":-}": "emotsmile",
        ":-": "emotneutral",
        ":-\\": "emotfrown",
        ":;": "emotneutral",
        "::": "emotneutral",
        ":/": "emotfrown",
        ":'|": "emotneutral",
        ":'/": "emotfrown",
        ":')": "emotsmile",
        ":'{": "emotfrown",
        ":'}": "emotsmile",
        ":'\\": "emotneutral",
        ":(": "emotfrown",
        ":)": "emotsmile",
        ":]": "emotsmile",
        ":[": "emotfrown",
        ":{": "emotfrown",
        ":}": "emotsmile",
        ":": "emotneutral",
        ":*{": "emotfrown",
        ":\\": "emotfrown",
        "(=": "emotsmile",
        "(;": "emotsmile",
        "(':": "emotsmile",
        ")=": "emotfrown",
        ")':": "emotfrown",
        "[;": "emotsmile",
        "]:": "emotfrown",
        "{:": "emotsmile",
        "\\=": "emotfrown",
        "\\:": "emotfrown",
        "<3": "emotheart"
}

for emot in emoticons.keys():
    EMOTICONS[emot] = emoticons[emot]

class TextPreprocessor:
    """
    This class is used to preprocess the text of the tweets.
    It removes the punctuation, stopwords, links and expands the contractions.

    Methods:
        - preprocess_text(text : str) -> str
    """
    
    def __init__(self):
        pass

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
                    # emoticons_str = emoticons_str.replace(emoticon, emoji.emojize(f'xx{EMOTICONS_EMO[emoticon].split("or")[0].split(",")[0].rstrip().replace(" ", "_").lower()} ')) # this line is used to convert emoticons to emojis
                    emoticons_str = emoticons_str.replace(emoticon, f' xx{EMOTICONS[emoticon].split("or")[0].split(",")[0].rstrip().replace(" ", "_").lower()} ')

        return emoticons_str
    

    def preprocess_text_old(self, text : str) -> str:
        """
        This function preprocesses the text.
        It removes the punctuation, stopwords, links and expands the contractions.
        Fruthermore, it lemmatizes the text and convert the emojis/emoticons.
        
        Args:
            - text {str}: the sentence to be processed
        Returns:
            - text {str}: the processed sentence
        """
        ## lowercase
        text = text.lower()
       
        ## Remove links
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S+|www\.\S+"
        text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

        ## Convert emoticons
        text = self.convert_emoticons(text)

        # ## Expand contractions
        text = contractions.fix(text)

        ## Lemmatize
        doc = self.nlp(text)
        tokens = [f':{token.lemma_.lower().strip().lstrip("x")}:' if token.lemma_.lower().strip().startswith("xx") else token.lemma_.lower().strip() for token in doc if not token.is_punct and not token.like_num]

        ## Convert emojis
        tokens = [self.convert_emojis(token) for token in tokens]

        return " ".join(tokens)
    
    def preprocess_text(self, text : str) -> str:
        """
        This function preprocesses the text.
        It removes the punctuation, stopwords, links and expands the contractions.
        Fruthermore, it lemmatizes the text and convert the emojis/emoticons.
        
        Args:
            - text {str}: the sentence to be processed
        Returns:
            - text {str}: the processed sentence
        """
        ## remove spaces at the beginning and at the end of the text
        text = text.strip()  
        # ## correct the spelling
        # text = self.tool.correct(text) 
        ## lowercase
        text = text.lower()
        ## substitute the user mentions with the word "xxuser"
        text = re.sub("<user>", "xxuser", text)
        ## substitute the links with the word "xxurl"
        text = re.sub("<url>", "xxurl", text)
        ## convert emoticons
        text = self.convert_emoticons(text)
        ## treat the hashtags as specialtokens
        text = text.replace("#", "xx")
        ## remove usless spaces
        text = re.sub("\\s+", " ", text)

        return text


## Test:
if __name__ == '__main__':
    # text = "I don't want to be a students 😍 :). I'm learnig NLP. @Filippo https://www.google.com  , Sup dude, wanna grab some www.youtube.com grub and chillax at the crib later?"
    text_2 = "<user> y dont follow me liam ? u can make me happy and make me feel better ( cuz im sick ) :(  <3 if u will follow me  #liamfollow me <url>"
    text_preprocessor = TextPreprocessor()
    # # preprocessed_text = text_preprocessor.preprocess_text_old(text)
    preprocessed_text_2 = text_preprocessor.preprocess_text(text_2)


    # print(preprocessed_text)
    print(preprocessed_text_2)
    from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print(tokenizer.tokenize(preprocessed_text_2))

