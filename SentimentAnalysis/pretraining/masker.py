import math
import random

import pandas as pd
import torch
from flashtext import KeywordProcessor
from transformers import PreTrainedTokenizerFast

from CONSTANTS import *


class Embedder:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, fraction_masking=FRACTION_MASKING,
                 lower_limit_fraction_masking=LOWER_LIMIT_FRACTION_MASKING):
        self.tokenizer = tokenizer
        self.mask_tosken_for_replacement = "<mask>"
        self.mask_token_id = self.tokenizer.get_vocab()[self.mask_tosken_for_replacement]
        self.sentiment_knowledge_kp = KeywordProcessor(case_sensitive=True)
        self.load_sentiment_knowledge_words()
        self.fraction_masking = fraction_masking
        self.lower_limit_fraction_masking = lower_limit_fraction_masking

    def load_sentiment_knowledge_words(self):
        df = pd.read_csv(PATH_POLARITY, dtype={"word": str, "polarity": float})
        # Filter by pmi value (keep pmi bigger than 5)
        df = df[abs(df.polarity) > 25.0]
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

        num_masks = math.floor(self.fraction_masking * available_tokens)
        num_masks_low = math.ceil(self.lower_limit_fraction_masking * available_tokens)

        # Randomly sample indexes of tokens that can be replaced
        sampled_indeces = []
        if len(indeces) >= num_masks:
            sampled_indeces += sorted(random.sample(indeces, num_masks))
        elif len(indeces) >= num_masks_low:
            sampled_indeces += sorted(random.sample(indeces, num_masks_low))
        else:
            sampled_indeces += indeces
            # Sample from remaining indeces
            remaining_indeces = set(range(1, len(tokens_with_replacement)
                                          - tokens_with_replacement.count(self.tokenizer.pad_token_id) - 1)) \
                                - set(indeces)
            sampled_indeces += random.sample(remaining_indeces,
                                             min(num_masks_low - len(indeces), len(remaining_indeces)))
            sampled_indeces.sort()

        attention_mask = encode_plus_res['attention_mask']
        masked_sentiment_words = 0
        additional_indeces = []
        ind_input_ids = 1
        limit = len(encode_plus_res['input_ids']) - encode_plus_res['input_ids'].count(self.tokenizer.pad_token_id) - 1

        for ind_indeces in sampled_indeces:
            if tokens_with_replacement[ind_indeces] == self.tokenizer.mask_token_id:
                masked_sentiment_words += 1
            else:
                while tokens_with_replacement[ind_indeces] != encode_plus_res['input_ids'][ind_input_ids] \
                        and encode_plus_res['input_ids'][ind_input_ids] != self.tokenizer.eos_token_id:
                    # Add index to list of indeces that can be sampled later
                    additional_indeces.append(ind_input_ids)
                    # Go to next input token
                    ind_input_ids += 1

                if ind_input_ids <= limit:
                    # Add mask for token
                    attention_mask[ind_input_ids] = 0
                    # Go to next input token
                    ind_input_ids += 1

        # Add masks to reach desired fraction of masks
        to_mask = random.sample(additional_indeces, masked_sentiment_words)
        for i in to_mask:
            attention_mask[i] = 0

        return {
            'ids': torch.tensor(encode_plus_res['input_ids'], dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long)
        }
