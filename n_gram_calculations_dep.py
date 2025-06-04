import pandas as pd 
import spacy
from tqdm import trange
from spacy_conll import ConllFormatter
from conllu import parse
from supar import Parser
import pickle 
import numpy as np
import math 
from minicons import scorer
from collections import defaultdict
import os 

# -----------------------------
# Utility Functions
# -----------------------------

def conllu2sen_modified(tokenlist) -> str:
    results =[x[i] for x in tokenlist for i in range(len(x))]
    tokens = [x['form'] for x in results]

    
    return ' '.join(tokens)


def conllu2sen(tokenlist) -> str:
    tokens = [x['form'] for x in tokenlist]
    
    return ' '.join(tokens)


def load_pickled_deps_list(dataset_name, lang, num_splits, base_dir='./parsed_datasets/training_datasets/pickle/'):
    docs = []

    for i in trange(num_splits, desc=f"Loading {dataset_name} {lang} dependencies"):
        file_path = f"{base_dir}{lang}_deps/{dataset_name}/conllu_deps_{i}.pickle"

        try:
            with open(file_path, 'rb') as f:
                split = pickle.load(f)

                for doc in split:
                    sen = conllu2sen(doc) ###maybe you need the modified version??
                    docs.append((sen, doc))
        
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping.")

    return docs



def load_pickled_deps_dict(dataset_name, paradigm_name, lang, num_splits, base_dir='./parsed_datasets/clams/pickle/'):
    docs = {}

    for i in trange(num_splits, desc=f"Loading {dataset_name} {lang} dependencies into dictionary"):
        file_path = f"{base_dir}{dataset_name}_{lang}_deps/{dataset_name}_{lang}_{paradigm_name}_deps_{i}.pickle"

        try:
            with open(file_path, 'rb') as f:
                split = pickle.load(f)

                for doc in split:
                    sen = conllu2sen(doc)
                    if sen not in docs:
                        docs[sen] = doc

        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping.")

    return docs


# -----------------------------
# Frequency Calculation
# -----------------------------

def process_docs_for_token_deprel(docs):
    """
    Processes a set of documents to count the frequency of each token paired with its dependency relation.
    
    Parameters:
        docs (dict): A dictionary where the keys are sentences and values are lists of tokens (each token is a dictionary with 'form' and 'deprel').
    
    Returns:
        pd.DataFrame: A DataFrame with 'form', 'deprel', and 'count' columns representing the frequency of each token-deprel pair.
    """
    # Initialize a defaultdict to count occurrences
    token_deprel_counts = defaultdict(int)
    token_counts = defaultdict(int)

    # Process each sentence in the documents
    for element in docs:
        sent = element[0]
        parse = element[1]

        for token in parse:
            form = token['form']
            deprel = token['deprel']
            token_deprel_counts[(form, deprel)] += 1
            token_counts[form] += 1
                

    # Convert the counts to a DataFrame
    data = {'form': [], 'deprel': [], 'count': []}
    for (form_deprel, count) in token_deprel_counts.items():
        form, deprel = form_deprel
        data['form'].append(form)
        data['deprel'].append(deprel)
        data['count'].append(count)

    df_token_deprel = pd.DataFrame(data)

    # Convert the total token counts to a DataFrame
    data_total = {'form': [], 'total_count': []}
    for form, total_count in token_counts.items():
        data_total['form'].append(form)
        data_total['total_count'].append(total_count)

    df_token_total = pd.DataFrame(data_total)
    
    return df_token_deprel, df_token_total




def process_docs_for_bigrams_deprel(docs):
    """
    Processes a set of documents to count the frequency of bigrams of token forms along with their corresponding dependency relations.
    
    Parameters:
        docs (dict): A dictionary where the keys are sentences and values are lists of tokens (each token is a dictionary with 'form' and 'deprel').
    
    Returns:
        pd.DataFrame: A DataFrame with 'form_bigram', 'deprel_bigram', and 'count' columns representing the frequency of form-deprel bigram pairs.
    """
    # Initialize defaultdict to count occurrences
    form_deprel_counts = defaultdict(int)

    # Process each sentence in the documents
    for element in docs:
        sent = element[0]  # Sentence text
        parse = element[1]  # Tokens (list of OrderedDict)
        prev_form = None
        prev_deprel = None

        # Process consecutive tokens to form pairs
        for token in parse:
            form = token.get('form', None)
            deprel = token.get('deprel', None)
            
            # We need to skip if form or deprel are missing
            if form is None or deprel is None:
                continue

            # If there's a previous form and deprel, count the pair with its corresponding deprel
            if prev_form is not None and prev_deprel is not None:
                form_deprel_counts[((prev_form, form), (prev_deprel, deprel))] += 1

            # Update previous form and deprel
            prev_form = form
            prev_deprel = deprel

    # Convert the counts to a DataFrame
    data = {'form_pair': [], 'deprel_pair': [], 'count': []}
    for (form_pair, deprel_pair), count in form_deprel_counts.items():
        form_pair_str = f"{form_pair[0]} -> {form_pair[1]}"
        deprel_pair_str = f"{deprel_pair[0]} -> {deprel_pair[1]}"
        data['form_pair'].append(form_pair_str)
        data['deprel_pair'].append(deprel_pair_str)
        data['count'].append(count)

    df_form_deprel = pd.DataFrame(data)
    
    return df_form_deprel




if __name__ == "__main__":
    language_of_interest = input("Please enter the language you prefer (eng, fr, de): ").strip().lower()

    # Validate the language input
    while language_of_interest not in ['eng', 'fr', 'de']:
        print("Invalid language. Please enter 'eng', 'fr', or 'de'.")
        language_of_interest = input("Please enter the language you prefer (english, french, german): ").strip().lower()

    for corpus_type in ["wiki", "childes"]:
        
        new_docs = load_pickled_deps_list(corpus_type, language_of_interest, 20)
        dep_by_deprel_df, dep_total_df  = process_docs_for_token_deprel(new_docs)
        dep_bigram = process_docs_for_bigrams_deprel(new_docs)
        dep_by_deprel_df.to_csv(f'./n_grams_frequency/{language_of_interest}/parsed_frequency_{corpus_type}/unigram_deprel_freq.csv', index=False)
        dep_bigram.to_csv(f'./n_grams_frequency/{language_of_interest}/parsed_frequency_{corpus_type}/bigrams_deprel_freq.csv', index=False)
