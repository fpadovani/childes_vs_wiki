from curses import def_prog_mode
import pandas as pd 
import os 
import json
import sys
from sympy import intersection
# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.variables import *
import spacy
import re 


def load_sentences_from_csv(file_path, column_name='sentences'):
    try:
        df = pd.read_csv(file_path)
        if column_name not in df:
            raise ValueError(f"Column '{column_name}' not found in {file_path}")
        # Drop missing values and ensure all entries are strings
        sentences = df[column_name].dropna().astype(str).tolist()
        print(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")


def tokenize_sentences(sentences):
    
    tokens_total = []
    for sentence in sentences:
        tokens = re.findall(r"\w+|[^\w\s]", sentence.lower())
        tokens_total.extend(tokens)
    return tokens


def tokenize_sentences_spacy(sentences, nlp):
    
    all_tokens = []
    for sentence in sentences:
        doc = nlp(sentence)
        tokens = [token.text for token in doc] 
        all_tokens.extend(tokens) 
    return all_tokens


def get_intersection(tokens1, tokens2):
    return set(tokens1).intersection(tokens2)


def lemmatize_tokens(tokens, nlp):
    nlp.max_length = max(len(" ".join(tokens)), nlp.max_length)

    chunk_size = 100000
    lemmatized_tokens = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        doc = nlp(" ".join(chunk))
        lemmatized_tokens.extend([token.lemma_ for token in doc])
    return lemmatized_tokens


def cache_lemmatized_tokens(tokens, cache_file):
    """
    Cache or retrieve lemmatized tokens.
    If a cached file exists, it loads and returns the lemmatized tokens.
    If not, it lemmatizes the input tokens, saves the result, and returns it.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            print(f"Loading cached lemmatized tokens from {cache_file}")
            return json.load(f)
    
    print(f"Lemmatizing tokens and caching to {cache_file}")
    lemmatized = set(lemmatize_tokens(tokens))
    
    with open(cache_file, 'w') as f:
        json.dump(lemmatized, f)
    
    return lemmatized

def get_intersection_with_lemmas(tokens1, tokens2):
    """
    Find the intersection of two token lists, considering lemmatized forms.
    """
    # Lemmatize both token sets
    lemmas1 = set(lemmatize_tokens(tokens1))
    lemmas2 = set(lemmatize_tokens(tokens2))
    return lemmas1.intersection(lemmas2)

def find_missing_words(file_path, intersection, is_blimp=False, is_clams=False, nlp=None):
    """
    Identify tokens in a file that are not present in the intersection set.
    """
    missing_words = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            if is_clams:
                sentence = line.split('\t')[-1].strip().split()
            else:
                sentence = line.split()

            # Tokenize and clean punctuation
            sentence_tokens = [token.lower() for token in sentence if token != ',' and token != '.']

            if is_blimp:
                sentence_tokens = [token.rstrip('.').rstrip('?').rstrip("'s").rstrip('!') for token in sentence_tokens]

            for token in sentence_tokens:
                if not token.strip():  # Skip empty tokens
                    continue
            
                doc = nlp(token)
                if len(doc) == 0:  # Ensure the doc is not empty
                    continue
            
                #lemma = doc[0].lemma_
                if token not in intersection: #and lemma not in intersection:
                    missing_words.append(token)

    return list(set(missing_words))


def process_paradigm_files(paradigm_folder, intersection, nlp):
    """
    Process all files in the paradigm folder and find missing tokens.
    """
    paradigms_missing = {}
    for file_name in os.listdir(paradigm_folder):
        file_path = os.path.join(paradigm_folder, file_name)
        missing_words = find_missing_words(file_path, intersection, is_blimp='blimp' in file_path, is_clams='clams' in file_path, nlp = nlp)
        paradigms_missing[file_name.rstrip('.txt')] = missing_words
    return paradigms_missing


def save_missing_words(paradigms_missing, output_file):
    """
    Save missing words dictionary to a JSON file.
    """
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(paradigms_missing, f, indent=4, ensure_ascii=False)
    print(f"Missing words saved to {output_file}")


def process_paradigm_files(paradigm_folder):
    """
    Process all files in the paradigm folder and find missing tokens.
    """
    #create new folder
    paradigm_folder_new = paradigm_folder + '_ok'
    if not os.path.exists(paradigm_folder_new):
        os.makedirs(paradigm_folder_new)

    paradigms_missing = {}
    for file_name in os.listdir(paradigm_folder):
        file_path = os.path.join(paradigm_folder, file_name)

        if 'excluded' in paradigm_folder:
            file_path_new = os.path.join(paradigm_folder_new, file_name)
            with open(file_path_new, 'w', encoding="utf-8") as f_writing:
            
                with open(file_path, 'r', encoding="utf-8") as f:
                    for line in f:
                        line_tokens = line.split()
                        line_new = [tok.rstrip('_').lstrip('_') for tok in line_tokens]
                        line_new_str = ' '.join(line_new)
                        f_writing.write(line_new_str + '\n')

        else:
            file_path_new = os.path.join(paradigm_folder_new, file_name)
            with open(file_path_new, 'w', encoding="utf-8") as f_writing:
            
                with open(file_path, 'r', encoding="utf-8") as f:
                    for line in f:
                        line_tokens = line.split()
                        line_new = line_tokens[1:]
                        line_new_str = ' '.join(line_new)
                        f_writing.write(line_new_str + '\n')

    return 


def filter_and_save_paradigms(paradigms_missing, excluded_folder, filtered_folder, paradigms_folder, is_clams=False):
    """
    Filter paradigms based on missing words, saving valid and excluded pairs into separate folders.
    """
    # Ensure the output folders exist
    os.makedirs(excluded_folder, exist_ok=True)
    os.makedirs(filtered_folder, exist_ok=True)

    for file_name in os.listdir(paradigms_folder):
        file_path = os.path.join(paradigms_folder, file_name)

        # Read the lines in the current paradigm file
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        # Get the missing words for this paradigm
        paradigm_name = file_name.rstrip('.txt')
        missing_words = paradigms_missing.get(paradigm_name, [])

        # Prepare filtered and excluded files
        filtered_lines = []
        excluded_lines = []

        # Process sentences in pairs
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):  # Skip incomplete pairs
                break

            # Parse the sentences in the pair
            sentence_1 = lines[i].strip()
            sentence_2 = lines[i + 1].strip()

            if is_clams:
                sentence_1 = " ".join(sentence_1.split()[1:])  
                sentence_2 = " ".join(sentence_2.split()[1:])  

            # Clean and tokenize both sentences, removing punctuation and lowering the case
            tokens_1 = [token.lower() for token in sentence_1.split() if token != ',' and token != '.']
            tokens_2 = [token.lower() for token in sentence_2.split() if token != ',' and token != '.']

            contains_missing_word_1 = any(word in missing_words for word in tokens_1)
            contains_missing_word_2 = any(word in missing_words for word in tokens_2)
            contains_missing_word = contains_missing_word_1 or contains_missing_word_2

            if contains_missing_word:
                # Highlight missing words in sentence_1
                highlighted_sentence_1 = " ".join(
                    f"_{token}_" if token in missing_words else token
                    for token in tokens_1
                )

                # Highlight missing words in sentence_2
                highlighted_sentence_2 = " ".join(
                    f"_{token}_" if token in missing_words else token
                    for token in tokens_2
                )

                excluded_lines.extend([highlighted_sentence_1 + "\n", highlighted_sentence_2 + "\n"])
            else:
                filtered_lines.extend([sentence_1.lower() + "\n", sentence_2.lower() + '\n'])  # Add both sentences to filtered

        # Save excluded sentences
        excluded_file_path = os.path.join(excluded_folder, file_name)
        with open(excluded_file_path, 'w', encoding="utf-8") as f:
            f.writelines(excluded_lines)

        # Save filtered sentences
        filtered_file_path = os.path.join(filtered_folder, file_name)
        with open(filtered_file_path, 'w', encoding="utf-8") as f:
            f.writelines(filtered_lines)

        print(f"Processed {file_name}: {len(filtered_lines)} sentences kept, {len(excluded_lines)} sentences excluded.")



def main():
    
    eng_nlp = spacy.load('en_core_web_sm')
    # Step 1: Load and tokenize CHILDES dataset
    childes_sentences = load_sentences_from_csv(AO_CHILDES_ENGLISH)
    childes_tokens = tokenize_sentences_spacy(childes_sentences, eng_nlp)

    # Step 2: Load and tokenize Wikipedia dataset 
    wiki_sentences = load_sentences_from_csv(WIKIPEDIA_ENG)
    wiki_tokens = tokenize_sentences_spacy(wiki_sentences, eng_nlp)

    intersection_eng = get_intersection(childes_tokens, wiki_tokens)
    print(f"Number of intersecting tokens for English: {len(intersection_eng)}")

    print("Processing Zorro paradigms...")
    zorro_paradigms_missing = process_paradigm_files(ZORRO_DIR, intersection_eng, nlp = eng_nlp)

    print("Processing BLiMP paradigms...")
    blimp_paradigms_missing = process_paradigm_files(BLIMP_DIR, intersection_eng, nlp = eng_nlp)

    #print("Processing CLAMS paradigms...")
    clams_paradigms_missing = process_paradigm_files(CLAMS_FOLDER_ENG, intersection_eng, nlp = eng_nlp)

    # Save results
    save_missing_words(zorro_paradigms_missing, MISSING_WORDS_ZORRO)
    save_missing_words(blimp_paradigms_missing, MISSING_WORDS_BLIMP)
    save_missing_words(clams_paradigms_missing, MISSING_WORDS_CLAMS_ENG)
    
    # Step 7: Filter paradigms and save filtered/excluded versions
    print("Filtering and saving paradigms...")



    filter_and_save_paradigms(
        paradigms_missing=zorro_paradigms_missing,
        excluded_folder=ZORRO_EXCLUDED_FOLDER,
        filtered_folder=ZORRO_FILTERED_FOLDER,
        paradigms_folder=ZORRO_DIR
    )

    filter_and_save_paradigms(
        paradigms_missing=blimp_paradigms_missing,
        excluded_folder=BLIMP_EXCLUDED_FOLDER,
        filtered_folder=BLIMP_FILTERED_FOLDER,
        paradigms_folder=BLIMP_DIR
    )
    

    filter_and_save_paradigms(
        paradigms_missing=clams_paradigms_missing,
        excluded_folder=CLAMS_EXCLUDED_FOLDER_ENG,
        filtered_folder=CLAMS_FILTERED_FOLDER_ENG,
        paradigms_folder=CLAMS_FOLDER_ENG, is_clams=True
    )
    ## FRENCH FILTERING 
    nlp_fr = spacy.load("fr_core_news_sm")
    childes_fr_sentences = load_sentences_from_csv(AO_CHILDES_FRENCH)
    childes_fr_tokens = tokenize_sentences_spacy(childes_fr_sentences, nlp_fr)

    # Step 2: Load and tokenize Wikipedia dataset
    wiki_fr_sentences = load_sentences_from_csv(WIKIPEDIA_FR)
    wiki_fr_tokens = tokenize_sentences_spacy(wiki_fr_sentences, nlp_fr) 

    intersection_fr = get_intersection(childes_fr_tokens, wiki_fr_tokens)
    print(f"Number of intersecting tokens: {len(intersection_fr)}")

    # Step 6: Process CLAMS paradigms
    print("Processing CLAMS paradigms...")
    clams_paradigms_missing_fr = process_paradigm_files(CLAMS_FOLDER_FR, intersection_fr, nlp = nlp_fr)

    save_missing_words(clams_paradigms_missing_fr, MISSING_WORDS_CLAMS_FR)

    # Step 7: Filter paradigms and save filtered/excluded versions
    print("Filtering and saving paradigms for French...")

    filter_and_save_paradigms(
        paradigms_missing=clams_paradigms_missing_fr,
        excluded_folder=CLAMS_EXCLUDED_FOLDER_FR,
        filtered_folder=CLAMS_FILTERED_FOLDER_FR,
        paradigms_folder=CLAMS_FOLDER_FR,
        is_clams=True
    )
    


    ## GERMAN FILTERING
    de_nlp = spacy.load("de_core_news_sm")
    childes_de_sentences = load_sentences_from_csv(AO_CHILDES_GERMAN)
    childes_de_tokens = tokenize_sentences_spacy(childes_de_sentences, de_nlp)

    # Step 2: Load and tokenize Wikipedia dataset
    wiki_de_sentences = load_sentences_from_csv(WIKIPEDIA_DE)
    wiki_de_tokens = tokenize_sentences_spacy(wiki_de_sentences, de_nlp) 

    intersection_de = get_intersection(childes_de_tokens, wiki_de_tokens)
    print(f"Number of intersecting tokens: {len(intersection_de)}")

    # Step 6: Process CLAMS paradigms
    print("Processing CLAMS paradigms...")
    clams_paradigms_missing_de = process_paradigm_files(CLAMS_FOLDER_DE, intersection_de, nlp = de_nlp)

    save_missing_words(clams_paradigms_missing_de, MISSING_WORDS_CLAMS_DE)

    # Step 7: Filter paradigms and save filtered/excluded versions
    print("Filtering and saving paradigms for German...")
        
    filter_and_save_paradigms(
        paradigms_missing=clams_paradigms_missing_de,
        excluded_folder=CLAMS_EXCLUDED_FOLDER_DE,
        filtered_folder=CLAMS_FILTERED_FOLDER_DE,
        paradigms_folder=CLAMS_FOLDER_DE,
        is_clams=True
    )



if __name__ == "__main__":
    main()
