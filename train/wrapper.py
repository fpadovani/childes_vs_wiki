import os
from typing import Optional, Tuple

from datasets import DatasetDict, disable_caching, load_dataset
from transformers import PreTrainedTokenizerFast


def break_into_chunks(input_list, chunk_length):
    for i in range(0, len(input_list), chunk_length):
        yield input_list[i : i + chunk_length]


def tokenize_wrapper(
    tokenizer,
    concat_all_sentences=False,
    minimal_sentence_length=0,
    maximal_sentence_length=128,
):
    def tokenize(element):
        if concat_all_sentences:
            sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id

            if None in element["text"]:
                combined_text = [text for text in element["text"] if text is not None]
                tokenized_sentences = tokenizer(combined_text)["input_ids"]
            
            else:
                tokenized_sentences = tokenizer(element["text"])["input_ids"]

            input_ids = []
            current_batch = []

            # Fill each batch up, and if it overflows start the next batch with the overflowing sentence.
            for sen in tokenized_sentences:
                batch_length_remaining = maximal_sentence_length - len(current_batch)
                if len(sen) > (batch_length_remaining - 1):
                    if batch_length_remaining > 4:
                        current_batch.extend(
                            [sep_token_id] + sen[: batch_length_remaining - 1]
                        )

                    input_ids.append(current_batch)

                    if len(sen) > maximal_sentence_length:
                        for current_batch in break_into_chunks(
                            sen, maximal_sentence_length
                        ):
                            if len(current_batch) == maximal_sentence_length:
                                input_ids.append(current_batch)
                                current_batch = []
                    else:
                        current_batch = sen
                else:
                    current_batch.extend([sep_token_id] + sen[:batch_length_remaining])

            if len(current_batch) > 0:
                input_ids.append(current_batch)

        else:
            input_ids = [
                item
                for item in tokenizer(element["text"])["input_ids"]
                if maximal_sentence_length > len(item) > minimal_sentence_length
            ]

        return {"input_ids": input_ids}

    return tokenize




def tokenize_wrapper_mlm(
    tokenizer,
    concat_all_sentences=False,
    minimal_sentence_length=0,
    maximal_sentence_length=128,
):
    def tokenize(element):
        if concat_all_sentences:
            sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id

            if None in element["text"]:
                combined_text = [text for text in element["text"] if text is not None]
                tokenized_sentences = tokenizer(combined_text)["input_ids"]
            
            else:
                tokenized_sentences = tokenizer(element["text"])["input_ids"]

            input_ids = []
            current_batch = []

            # Fill each batch up, and if it overflows start the next batch with the overflowing sentence.
            for sen in tokenized_sentences:
                batch_length_remaining = maximal_sentence_length - len(current_batch)
                if len(sen) > (batch_length_remaining - 1):
                    if batch_length_remaining > 4:
                        current_batch.extend(
                            [sep_token_id] + sen[: batch_length_remaining - 1]
                        )
                    current_batch += [tokenizer.pad_token_id] * (maximal_sentence_length - len(current_batch))
                    input_ids.append(current_batch)

                    if len(sen) > maximal_sentence_length:
                        for current_batch in break_into_chunks(
                            sen, maximal_sentence_length
                        ):
                            if len(current_batch) == maximal_sentence_length:
                                input_ids.append(current_batch)
                                current_batch = []
                    else:
                        current_batch = sen
                else:
                    current_batch.extend([sep_token_id] + sen[:batch_length_remaining])

            if len(current_batch) > 0:
                current_batch += [tokenizer.pad_token_id] * (maximal_sentence_length - len(current_batch))
                input_ids.append(current_batch)

        else:
            input_ids = [
                item
                for item in tokenizer(element["text"])["input_ids"]
                if maximal_sentence_length > len(item) > minimal_sentence_length
            ]
        return {"input_ids": input_ids}

    return tokenize



def add_attention_mask_and_labels(example, pad_token_id):
    """
    Adds attention_mask and labels to a single example with input_ids.
    
    Args:
        example (dict): A dictionary containing `input_ids`.
        pad_token_id (int): The tokenizer's pad token ID.
        
    Returns:
        dict: Updated example with attention_mask and labels.
    """
    input_ids = example["input_ids"]
    
    # Generate attention mask (1 for non-pad tokens, 0 for pad tokens)
    #attention_mask = [1 if token != pad_token_id else 0 for token in input_ids]
    
    # Labels are a copy of input_ids
    labels = input_ids.copy()
    
    #example["attention_masks"] = attention_mask
    example["labels"] = labels
    return example