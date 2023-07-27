import math
import random

import pandas as pd
import torch
from flashtext import KeywordProcessor
from transformers import PreTrainedTokenizerFast

from CONSTANTS import *


class Masker:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, fraction_masking=FRACTION_MASKING,
                 lower_limit_fraction_masking=LOWER_LIMIT_FRACTION_MASKING, twitter=True):
        self.tokenizer = tokenizer
        self.mask_tosken_for_replacement = MASK_TOKEN_FOR_REPLACEMENT
        self.mask_token_id = self.tokenizer.get_vocab()[self.mask_tosken_for_replacement]
        self.sentiment_knowledge_kp = KeywordProcessor(case_sensitive=True)
        self.load_sentiment_knowledge_words(twitter)
        self.fraction_masking = fraction_masking
        self.lower_limit_fraction_masking = lower_limit_fraction_masking

    def load_sentiment_knowledge_words(self, twitter):
        if twitter:
            df = pd.read_csv(PATH_POLARITY_TWITTER, dtype={"word": str, "polarity": float})
            threshold = TWITTER_THRESHOLD
        else:
            df = pd.read_csv(PATH_POLARITY_AMAZON, dtype={"word": str, "polarity": float})
            threshold = AMAZON_THRESHOLD

        # Filter by pmi value
        df = df[abs(df.polarity) > threshold]
        # Insert words in keyword processor.
        for row in df.itertuples():
            self.sentiment_knowledge_kp.add_keyword(row.word, self.mask_tosken_for_replacement)

    def encode_plus(self, text, add_special_tokens=True, max_length=MAX_LENGTH, return_attention_mask=True):
        # Get tokens id for string with sentiment words
        tokens_with_replacement = self.tokenizer.encode(self.sentiment_knowledge_kp.replace_keywords(text),
                                                        None,
                                                        add_special_tokens=add_special_tokens,
                                                        padding='max_length',
                                                        max_length=max_length,
                                                        truncation=True)

        indeces = []
        for i in range(1, len(tokens_with_replacement)):
            if tokens_with_replacement[i] == self.tokenizer.eos_token_id:
                break
            if tokens_with_replacement[i] != self.mask_token_id:
                indeces.append(i)

        # Get tokens_ids, attention_mask and token_type_ids
        encode_plus_res = self.tokenizer.encode_plus(text,
                                                     None,
                                                     add_special_tokens=add_special_tokens,
                                                     padding='max_length',
                                                     max_length=max_length,
                                                     return_attention_mask=return_attention_mask,
                                                     truncation=True)

        # Count padding tokens. "-2" accounts for cls and eos tokens
        available_tokens = len(encode_plus_res['input_ids']) - \
                           encode_plus_res['input_ids'].count(self.tokenizer.pad_token_id) - 2

        num_masks = math.ceil(self.fraction_masking * available_tokens)
        num_masks_low = math.ceil(self.lower_limit_fraction_masking * available_tokens)

        # Randomly sample indexes of tokens that can be replaced
        sampled_indeces = []
        if len(indeces) >= num_masks:
            sampled_indeces += sorted(random.sample(indeces, num_masks))
        elif len(indeces) >= num_masks_low:
            sampled_indeces += sorted(random.sample(indeces, num_masks_low))
        else:
            # Take available indeces
            sampled_indeces = indeces
            sampled_indeces.sort()

        attention_mask = encode_plus_res['attention_mask']
        ind_input_ids = 1
        limit = len(encode_plus_res['input_ids']) - encode_plus_res['input_ids'].count(self.tokenizer.pad_token_id) - 1

        for ind_indeces in sampled_indeces:
            while tokens_with_replacement[ind_indeces] != encode_plus_res['input_ids'][ind_input_ids] \
                        and ind_input_ids < limit:
                # Go to next input token
                ind_input_ids += 1

            if ind_input_ids < limit:
                # Add mask for token
                attention_mask[ind_input_ids] = 0
                # Go to next input token
                ind_input_ids += 1

        return {
            'input_ids': torch.tensor(encode_plus_res['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
