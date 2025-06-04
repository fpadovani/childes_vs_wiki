from datasets import Dataset, DatasetDict, load_from_disk
import random
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
import os
from datasets import load_dataset
from pathlib import Path
from typing import Optional
import numpy as np
import shutil
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import json
from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import WordLevel
from utils.variables import *
from datasets import load_dataset
import nltk
nltk.download('punkt')


def train_unified_tokenizer(dataset_files, save_path, vocab_size=8192, min_frequency=2):

    tokenizer = ByteLevelBPETokenizer()

    # Train tokenizer on dataset
    tokenizer.train(
        files=dataset_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<s>", "</s>", "<pad>", "<unk>", "<mask>", 
            "<|endoftext|>" 
        ]
    )

    # Save tokenizer
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_model(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    # Reload tokenizer in Hugging Face format
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        save_path,
        model_max_length=128,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        additional_special_tokens=["<|endoftext|>"])

    return tokenizer



def generate_wiki(base_dir, df_type, language):

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    file_path_1 = os.path.join(CORPORA_DIR, 'english', 'wikipedia', 'wikipedia1.txt')
    with open(file_path_1, 'r') as f:
        sentences = [line.strip() for line in f]

    max_tokens = 4_264_184
    selected_sentences = []
    token_counts = []
    total_tokens = 0
    seen_sentences = set()  # To track duplicates

    for sentence in sentences:
        # Skip empty or duplicate sentences
        if not sentence or sentence in seen_sentences:
            continue

        seen_sentences.add(sentence)  # Mark as seen immediately

        # Process sentence for token count
        doc = nlp(sentence)
        token_count = len([t for t in doc if not t.is_space and not t.is_punct])

        # Check if we’re still under the token limit
        if total_tokens + token_count > max_tokens:
            break

        selected_sentences.append(sentence)
        token_counts.append(token_count)
        total_tokens += token_count

    # Create a DataFrame
    df = pd.DataFrame({
        "sentences": selected_sentences,
        "token_count": token_counts
    })
    df.to_csv(os.path.join(CORPORA_DIR, 'english', 'wikipedia', 'wikipedia_final.csv'), index=False)

    # Validation set target: 8% of total tokens
    validation_token_target = int(total_tokens * 0.08)

    # Shuffle dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    val_indices = []
    val_token_count = 0

    for idx, row in df_shuffled.iterrows():
        if val_token_count >= validation_token_target:
            break
        val_indices.append(idx)
        val_token_count += row["token_count"]

    # Create splits
    evaluation_set = df_shuffled.iloc[val_indices]
    train_set = df_shuffled.drop(index=val_indices)

    # HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_set["sentences"].tolist()})
    valid_dataset = Dataset.from_dict({"text": evaluation_set["sentences"].tolist()})

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset
    })

    # Save to disk
    save_path = os.path.join(base_dir, language, df_type)
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. Removing existing folder.")
        shutil.rmtree(save_path)

    os.makedirs(save_path, exist_ok=True)
    train_file_path = os.path.join(save_path, 'train.csv')
    validation_file_path = os.path.join(save_path, 'validation.csv')

    train_dataset.to_csv(train_file_path)
    valid_dataset.to_csv(validation_file_path)

    return dataset_dict, train_file_path, validation_file_path



def save_dataset_to_disk(base_dir, df_type, train_set, validation_context, validation_random, language:Optional[str]):

    # Prepare the datasets
    train_dataset = Dataset.from_dict({
       "text": train_set['sentences'],
        "transcript_id": train_set['transcript_id'],
        "age_in_months": train_set['age_in_months'],
        "bucket": train_set['bucket']
    })

    valid_context = Dataset.from_dict({
        "text": validation_context['sentences'],
        "transcript_id": validation_context['transcript_id'],
        "age_in_months": validation_context['age_in_months'],
        "bucket": validation_context['bucket']
    })

    valid_random = Dataset.from_dict({
        "text": validation_random['sentences'],
        "transcript_id": validation_random['transcript_id'],
        "age_in_months": validation_random['age_in_months'],
        "bucket": validation_random['bucket']
    })

    dataset_dict = DatasetDict({'train': train_dataset, 'valid_in_context': valid_context, 'valid_random': valid_random})


    if language:
        save_path = os.path.join(base_dir,language, df_type)
    else:
        save_path = os.path.join(base_dir, df_type)
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. Removing existing folder.")
        shutil.rmtree(save_path)

    # Save the datasets to disk
    os.makedirs(save_path, exist_ok=True)
    train_file_path = os.path.join(save_path, 'train.csv')
    validation_in_context_path = os.path.join(save_path, 'validation_in_context.csv')
    validation_random_path = os.path.join(save_path, 'validation_random.csv')

    # Save datasets to CSV files
    train_dataset.to_csv(train_file_path)
    valid_context.to_csv(validation_in_context_path)
    valid_random.to_csv(validation_random_path)

    # Save the DatasetDict
    dataset_dict.save_to_disk(save_path)

    return dataset_dict, train_file_path, validation_in_context_path, validation_random_path




def generate_aochildes_single_language(input_file,base_dir, df_type, language):
    random.seed(42)

    if language == 'english':
        df = pd.read_csv(input_file)
    elif language == 'french':
        df = pd.read_csv(input_file)
    elif language == 'german':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    # Calculate 3% of total transcripts, ensuring at least 1 transcript
    total_transcript_ids = df['transcript_id'].unique()
    num_transcript_for_evaluation = max(1, int(len(total_transcript_ids) * 0.10))

    sampled_transcript_ids = random.sample(list(total_transcript_ids), num_transcript_for_evaluation)
    preliminary_context_set = df[df['transcript_id'].isin(sampled_transcript_ids)]
    preliminary_randomized_set = preliminary_context_set.sample(n=len(preliminary_context_set), random_state=42)
    sampled_indexes = preliminary_context_set.index.tolist()

    # Create the training set by excluding the sampled indexes
    train_set = df.drop(index=sampled_indexes).reset_index(drop=True)

    # Remove duplicates from evaluation sets that are present in train set
    preliminary_context_set = preliminary_context_set[
        ~preliminary_context_set["sentences"].isin(train_set["sentences"])
    ]

    preliminary_randomized_set = preliminary_randomized_set[
        ~preliminary_randomized_set["sentences"].isin(train_set["sentences"])
    ]


    # Save the dataset to disk
    dataset_dict, train_file_path, validation_in_context_path, validation_random_path = save_dataset_to_disk(
        base_dir, df_type, train_set, preliminary_context_set, preliminary_randomized_set, language=language
    )

    return dataset_dict, train_file_path, validation_in_context_path, validation_random_path



def process_wiki_dataset(input_file, language, nlp_model, dataset_name, token_limit, df_type, base_dir):
    nlp = spacy.load(nlp_model)
    total_tokens = 0
    selected_sentences = []
    token_counts = []
    seen_sentences = set()

    if language == 'german':
        dataset = load_dataset(dataset_name, trust_remote_code=True, split='train')
        for sentence in dataset['text']:
            sentence_cleaned = sentence.strip('"').lower()
            if not sentence_cleaned or sentence_cleaned in seen_sentences:
                continue

            seen_sentences.add(sentence_cleaned)
            doc = nlp(sentence_cleaned)
            tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
            sentence_token_count = len(tokens)

            if total_tokens + sentence_token_count > token_limit:
                break

            selected_sentences.append(sentence_cleaned)
            token_counts.append(sentence_token_count)
            total_tokens += sentence_token_count

    elif language == 'french':
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        train_data = dataset["train"]
        for paragraph in train_data['paragraph']:
            sentences = list(set(nltk.sent_tokenize(paragraph, language=language)))
            for sentence in sentences:
                sentence_cleaned = sentence.strip('"').lower()
                if not sentence_cleaned or sentence_cleaned in seen_sentences:
                    continue

                seen_sentences.add(sentence_cleaned)
                doc = nlp(sentence_cleaned)
                tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
                sentence_token_count = len(tokens)

                if total_tokens + sentence_token_count > token_limit:
                    break

                selected_sentences.append(sentence_cleaned)
                token_counts.append(sentence_token_count)
                total_tokens += sentence_token_count

            if total_tokens >= token_limit:
                break

    # Create DataFrame
    df = pd.DataFrame({
        "text": selected_sentences,
        "token_count": token_counts
    })

    # Save raw collected data if needed
    df.to_csv(input_file, index=False)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    validation_token_target = int(total_tokens * 0.08)

    val_indices = []
    val_token_count = 0

    for idx, row in df_shuffled.iterrows():
        if val_token_count >= validation_token_target:
            break
        val_indices.append(idx)
        val_token_count += row["token_count"]


    # Create splits
    evaluation_set = df_shuffled.iloc[val_indices]
    train_set = df_shuffled.drop(index=val_indices)

    # Save datasets
    save_path = os.path.join(base_dir, language, df_type)
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. Removing existing folder.")
        shutil.rmtree(save_path)

    os.makedirs(save_path, exist_ok=True)

    train_file_path = os.path.join(save_path, 'train.csv')
    validation_file_path = os.path.join(save_path, 'validation.csv')

    train_set[["text"]].to_csv(train_file_path, index=False)
    evaluation_set[["text"]].to_csv(validation_file_path, index=False)

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_set["text"].tolist()})
    valid_dataset = Dataset.from_dict({"text": evaluation_set["text"].tolist()})

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset
    })


    return dataset_dict, train_file_path, validation_file_path


def generate_wiki_fr(input_file,base_dir, df_type, language):
    return process_wiki_dataset(
        input_file,
        language='french',
        nlp_model='fr_core_news_sm',
        dataset_name='asi/wikitext_fr',
        token_limit=2287787,  # French dataset token limit
        df_type=df_type,
        base_dir=base_dir
    )

def generate_wiki_de(input_file,base_dir, df_type, language):
    return process_wiki_dataset(
        input_file,
        language='german',
        nlp_model='de_core_news_sm',
        dataset_name="gwlms/dewiki-20230701-nltk-corpus",
        token_limit=3733833,  # German dataset token limit
        df_type=df_type,
        base_dir=base_dir
    )



def create_dataset_splits_2(input_file, base_dir:str, df_type:str, language: Optional[str] = None):
    print("Let's create the dataset splits")

    if df_type == 'wikipedia':
        dataset_dict, train_file_path, valid_file_path = generate_wiki(base_dir, df_type, language)
        return dataset_dict, train_file_path, valid_file_path, None
    
    elif df_type == 'wikipedia_fr':
        dataset_dict, train_file_path, validation_in_context_path = generate_wiki_fr(input_file,base_dir, df_type, language)
        return dataset_dict, train_file_path, validation_in_context_path, None
        
    elif df_type == 'wikipedia_de':
        dataset_dict, train_file_path, validation_in_context_path = generate_wiki_de(input_file,base_dir, df_type, language)
        return dataset_dict, train_file_path, validation_in_context_path, None

    elif df_type == 'random':
        dataset_dict, train_set, preliminary_context, preliminary_random = generate_aochildes_single_language(input_file,base_dir, df_type, language)
        return dataset_dict, train_set, preliminary_context, preliminary_random

def scrumble_dataset(raw_datasets: DatasetDict, num_epochs, scramble_unit, base_dir:str, df_type:str, language: Optional[str] = None):
    '''
    Scrumble the dataset
    :param dataset_dict: dataset which contains the train, eval, test splits
    :return: dataset_dict_shuffled: the shuffled dataset
    '''

    train_split = raw_datasets['train'].to_pandas() 

    dataset_dict_shuffled = {}
    combined_transcripts = []  
    combined_sentences = []  
    for i in range(num_epochs):

        if scramble_unit == 'transcript': #this condition is just a shuffling, it's not breaking any coherence
            
            shuffled_buckets = []  # Collect the shuffled data for each bucket
            
            # Group by 'bucket'
            for bucket, group in train_split.groupby('bucket', sort=False):
                print(f"Processing bucket {bucket}...")

                transcript_ids = group['transcript_id'].unique()

                random.seed(i)  
                shuffled_transcript_ids = random.sample(list(transcript_ids), len(transcript_ids))
                shuffled_transcripts = [group[group['transcript_id'] == tid] for tid in shuffled_transcript_ids]

                shuffled_bucket_data = pd.concat(shuffled_transcripts)
                shuffled_buckets.append(shuffled_bucket_data)

            shuffled_train_set = pd.concat(shuffled_buckets).reset_index(drop=True)
            dataset_dict_shuffled[f'shuffled_epoch_{i}'] = Dataset.from_pandas(shuffled_train_set)
            combined_transcripts.append(shuffled_train_set)


        elif scramble_unit == 'sentence': #this condition is breaking the local coherence

            if df_type == 'random': 
                shuffled_train_set = train_split.sample(frac=1, random_state=i)
                dataset_dict_shuffled[f'shuffled_epoch_{i}'] = Dataset.from_pandas(shuffled_train_set)
                combined_sentences.append(shuffled_train_set)
            
            else:

                shuffled_buckets = []  
                for buck, group in train_split.groupby('bucket', sort=False):
                    shuffled_sentences = group.sample(frac=1, random_state=i)
                    shuffled_buckets.append(shuffled_sentences)
            

                shuffled_train_set = pd.concat(shuffled_buckets).reset_index(drop=True)
                dataset_dict_shuffled[f'shuffled_epoch_{i}'] = Dataset.from_pandas(shuffled_train_set)
                combined_sentences.append(shuffled_train_set)

    
    # Combine all the local coherence datasets if applicable
    if combined_transcripts:
        combined_t_df = pd.concat(combined_transcripts).reset_index(drop=True)
        combined_t_dataset = Dataset.from_pandas(combined_t_df)
    else:
        combined_t_dataset = None

    # Combine all the global coherence datasets if applicable
    if combined_sentences:
        
        combined_s_df = pd.concat(combined_sentences).reset_index(drop=True)
        combined_s_dataset = Dataset.from_pandas(combined_s_df)

        expected_size = len(raw_datasets['train']) * num_epochs
        actual_size =len(combined_s_dataset)
        print(f"Expected size: {expected_size}, Actual size: {actual_size}")
        
    else:
        combined_s_dataset = None
    if language:
        save_path = os.path.join(base_dir, language, df_type, 'epochs', str(num_epochs))
    else:
        save_path = os.path.join(base_dir, df_type, scramble_unit, 'epochs', str(num_epochs))
    os.makedirs(save_path, exist_ok=True)
    train_file_path = os.path.join(save_path, 'train.csv')
    validation_in_context_path = os.path.join(save_path, 'validation_in_context.csv')
    validation_random_path = os.path.join(save_path, 'validation_random.csv')

    # Create a new DatasetDict and add the combined datasets
    dataset_dict_shuffled = DatasetDict()
    if combined_t_dataset:
        dataset_dict_shuffled['train'] = combined_t_dataset
        combined_t_dataset.to_csv(train_file_path)

    if combined_s_dataset:
        dataset_dict_shuffled['train'] = combined_s_dataset
        combined_s_dataset.to_csv(train_file_path)

    raw_datasets['validation_ctx'].to_pandas().to_csv(validation_in_context_path, index = False)
    raw_datasets['validation_rnd'].to_pandas().to_csv(validation_random_path, index=False)

    print("Dataset scrambling and combination complete.")
    return dataset_dict_shuffled, train_file_path, validation_in_context_path, validation_random_path


def scramble_wikipedia(raw_datasets, num_epochs, dataset_folder, order, language: Optional[str] = None):
    """Scramble the Wikipedia dataset for multiple epochs."""
    print(f"Scrambling Wikipedia dataset for {num_epochs} epochs...")

    all_train_data = []

    for epoch in range(num_epochs):
    
        shuffled_dataset = raw_datasets["train"].shuffle(seed=epoch)
        all_train_data.extend(shuffled_dataset)

    if language:
        base_dir = os.path.join(dataset_folder, language, order, 'epochs', str(num_epochs))
    else:
        base_dir = os.path.join(dataset_folder, order, 'epochs', str(num_epochs))

    train_path = os.path.join(base_dir, 'train.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    pd.DataFrame(all_train_data).to_csv(train_path, index=False)
    valid_path = os.path.join(base_dir, 'validation.csv')
    raw_datasets['validation'].to_pandas().to_csv(valid_path, index=False)

    return train_path, valid_path
    
def load_existing_dataset(dataset_folder, order):
    """Load an existing dataset from the given folder."""
    dataset_path = Path(dataset_folder) / order
    print(f"Dataset folder '{dataset_path}' exists.")

    # Define data_files based on the 'order'
    data_files = {'train': str(dataset_path / 'train.csv')}
    
    if 'wikipedia' in order or 'french/wikipedia_fr' in order or 'german/wikipedia_de' in order:
        data_files['validation'] = str(dataset_path / 'validation.csv')

    else:
        data_files['validation_ctx'] = str(dataset_path / 'validation_in_context.csv')
        data_files['validation_rnd'] = str(dataset_path / 'validation_random.csv')

    # Use load_dataset and pass data_files as a dictionary with proper paths
    return load_dataset('csv', data_files=data_files)



def create_and_load_new_dataset(input_file, dataset_folder, order, save_fig, fig_folder, language: Optional[str] = None):
    """Create a new dataset split and load it."""
    print("Dataset folder doesn't exist, creating a new one.")

    dataset_dict, train_path, valid_path, valid_path1 = create_dataset_splits_2(input_file,base_dir=dataset_folder, df_type=order, language=language)

    data_files = {'train': train_path}
    if valid_path1 is None:  
        data_files['validation'] = valid_path
    else: 
        data_files['validation_ctx'] = valid_path
        data_files['validation_rnd'] = valid_path1

    return load_dataset('csv', data_files=data_files)

def loading(train_path, valid_ctx_path, valid_rnd_path:Optional[str], streaming:Optional[bool] = True):
    if valid_rnd_path:
        return load_dataset('csv', data_files={
            'train': train_path,
            'validation_ctx': valid_ctx_path,
            'validation_rnd': valid_rnd_path
        }, streaming=streaming)
    else:
        return load_dataset('csv', data_files={
            'train': train_path,
            'validation': valid_ctx_path
        }, streaming=streaming)
    


def handle_dataset_scrambling(raw_datasets, order, num_train_epochs, dataset_folder, scramble_unit, streaming=True, language: Optional[str] = None):
    """Handle scrambling and loading datasets based on order and epochs."""

    if order not in ['wikipedia', 'wikipedia_fr', 'wikipedia_de']:

        if num_train_epochs > 1 or (num_train_epochs == 1 and scramble_unit == 'sentence'):
            dataset_dict,train_path, valid_ctx_path, valid_rnd_path = scrumble_dataset(
                raw_datasets, num_train_epochs, scramble_unit, dataset_folder, order, language
            )
            return loading(train_path, valid_ctx_path, valid_rnd_path, streaming)
        else:
            train_path = os.path.join(dataset_folder, language, order, 'train.csv')
            valid_ctx_path = os.path.join(dataset_folder, language, order, 'validation_in_context.csv')
            valid_rnd_path = os.path.join(dataset_folder, language, order, 'validation_random.csv')
            return loading(train_path, valid_ctx_path, valid_rnd_path, streaming)

        
    elif order in ['wikipedia', 'wikipedia_fr', 'wikipedia_de'] or language: 
        if num_train_epochs > 1:
            train_path, valid_path = scramble_wikipedia(raw_datasets, num_train_epochs, dataset_folder, order, language)
            return loading(train_path, valid_path, None, streaming)
        
        else:
            # Handle cases where we only have a single epoch
            dataset_path = os.path.join(dataset_folder, language, order)
            train_path = os.path.join(dataset_path, 'train.csv')
            valid_path = os.path.join(dataset_path, 'validation.csv')
            return loading(train_path, valid_path, None, streaming)