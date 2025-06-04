import pandas as pd
from collections import Counter
import spacy
import csv
import os
from utils.variables import TRAINING_CHILDES_DE, TRAINING_CHILDES_ENG, TRAINING_CHILDES_FR, TRAINING_WIKI_DE, TRAINING_WIKI_ENG, TRAINING_WIKI_FR

# Function to save n-gram frequencies
def save_ngram_frequencies(language, unigram_freq, bigram_freq, folder):
    """ Saves unigram and bigram frequencies to CSV files. """
    output_dir_uni = f"./n_grams_frequency/{language}/{folder}/unigram"
    os.makedirs(output_dir_uni, exist_ok=True)

    output_dir_big = f"./n_grams_frequency/{language}/{folder}/bigram"
    os.makedirs(output_dir_big, exist_ok=True)
    
    # Save unigram frequencies
    with open(f"{output_dir_uni}/unigram_freq_{folder}_training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])
        writer.writerows(unigram_freq.items())
    
    # Save bigram frequencies
    with open(f"{output_dir_big}/bigram_freq_{folder}_training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "count"])
        writer.writerows([(w1, w2, count) for (w1, w2), count in bigram_freq.items()])




def load_spacy_model(language):
    """ Load spaCy model based on the input language. """
    if language == 'eng':
        return spacy.load('en_core_web_sm', disable=['parser', 'ner'])  
    elif language == 'fr':
        return spacy.load('fr_core_news_sm', disable=['parser', 'ner'])  
    elif language == 'de':
        return spacy.load('de_core_news_sm', disable=['parser', 'ner'])  
    else:
        raise ValueError("Language not supported. Choose from 'english', 'french', 'german'.")
    



# Function to get the appropriate dataset based on the language
def get_dataset(language):
    """ Return dataset file paths for both Childes and Wikipedia based on the input language. """
    if language == 'eng':
        return TRAINING_CHILDES_ENG, TRAINING_WIKI_ENG
    elif language == 'fr':
        return TRAINING_CHILDES_FR, TRAINING_WIKI_FR
    elif language == 'de':
        return TRAINING_CHILDES_DE, TRAINING_WIKI_DE
    else:
        raise ValueError("Language not supported. Choose from 'english', 'french', 'german'.")
    



# Function to calculate n-gram frequencies
def calculate_ngram_frequencies(file_path, nlp):
    """ Calculates unigram and bigram frequencies using spaCy tokenizer. """
    df = pd.read_csv(file_path)  # Read only first column
    unigram_counter = Counter()
    bigram_counter = Counter()
    
    for sentence in df['text']: 
        doc = nlp(sentence.lower())  # Tokenize sentence
        words_clean = [token.text for token in doc if token.is_alpha]  # Keep only alphabetic tokens
        
        unigram_counter.update(words_clean)  
        bigram_counter.update(zip(words_clean, words_clean[1:]))  # Update bigram frequency
    
    return dict(unigram_counter), dict(bigram_counter)




def process_language_data(language):
    """ Process the language data to calculate n-grams for Childes and Wikipedia. """
    
    # Load spaCy model based on language
    nlp = load_spacy_model(language)
    
    # Get dataset file paths for the selected language
    childes_file, wikipedia_file = get_dataset(language)
    
    # Calculate n-gram frequencies for Childes data
    print(f"Processing n-grams for {language} Childes data...")
    unigram_freq_childes, bigram_freq_childes = calculate_ngram_frequencies(childes_file, nlp)
    print(f"Saving n-gram frequencies for {language}...")
    save_ngram_frequencies(language, unigram_freq_childes, bigram_freq_childes, folder = 'childes')
    
    # Calculate n-gram frequencies for Wikipedia data
    print(f"Processing n-grams for {language} Wikipedia data...")
    unigram_freq_wikipedia, bigram_freq_wikipedia = calculate_ngram_frequencies(wikipedia_file, nlp)
    print(f"Saving n-gram frequencies for {language}...")
    save_ngram_frequencies(language, unigram_freq_wikipedia, bigram_freq_wikipedia, folder = 'wikipedia')
    
    print(f"Processing for {language} completed!")


if __name__ == "__main__":
    language_of_interest = input("Please enter the language you prefer (eng, fr, de): ").strip().lower()

    # Validate the language input
    while language_of_interest not in ['eng', 'fr', 'de']:
        print("Invalid language. Please enter 'eng', 'fr', or 'de'.")
        language_of_interest = input("Please enter the language you prefer (english, french, german): ").strip().lower()

    # Process the data based on the selected language
    process_language_data(language_of_interest)