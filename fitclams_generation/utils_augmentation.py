import pandas as pd 

def get_articles(lang):
    """
    Helper function to return article structure based on language.
    """
    if lang == 'en':
        return {
            'subject': "the ", 
            'object': "the ",  
            'subject_plur': "the ",  
            'object_plur': "the ",   
            'genitive_sing': "the ",
            'genitive_plur': "the "
        }
    
    elif lang == 'fr':
        return {
            'subject': lambda word, gender: "l'" if word[0].lower() in ['a', 'e', 'i', 'o', 'u', 'h'] else ('le ' if gender == 'Masc' else 'la '),
            'object': lambda word, gender: "l'" if word[0].lower() in ['a', 'e', 'i', 'o', 'u', 'h'] else ('le ' if gender == 'Masc' else 'la '),
            'subject_plur': "les ",  
            'object_plur': "les ",   
            'genitive_sing': lambda word, gender: "de l'" if word[0].lower() in ['a', 'e', 'i', 'o', 'u', 'h'] else ('du ' if gender == 'Masc' else 'de la '),
            'genitive_plur': "des "
    }
    
    elif lang == 'de':
        return {
            'subject': lambda word, gender: 'der ' if gender == 'Masc' else ('die ' if gender == 'Fem' else 'das '),
            'object': lambda word, gender: 'der ' if gender == 'Masc' else ('die ' if gender == 'Fem' else 'das '),
            'subject_plur': "die ",  
            'object_plur': "die ",   
            'genitive_sing': "dem ",  ## this is a dative, not a genitive
            'genitive_plur': "den ",
            'relative_pron_sing_nominative': lambda word,gender: 'der ' if gender == 'Masc' else ('die ' if gender == 'Fem' else 'das '),
            'relative_pron_sing_accusative': lambda word,gender: 'den ' if gender == 'Masc' else ('die ' if gender == 'Fem' else 'das '),
        }
    
    else:
        raise ValueError(f"Unsupported language: {lang}")


def load_data_by_option(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, opt, lang):
    '''if opt == 'entire':
        return mp_load_data(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, lang)'''
    if opt == 'wiki':
        return mp_load_partial_data(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, 'wiki')
    elif opt == 'childes':
        return mp_load_partial_data(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, 'childes')
    else:
        raise ValueError(f"Unsupported option: {opt}")
    
    

def merge_and_remove_duplicates(data_childes, data_wikipedia, on_columns):
    common_data = pd.merge(data_childes, data_wikipedia, on=on_columns, how='inner')
    data_wikipedia_filtered = data_wikipedia[~data_wikipedia[on_columns].isin(common_data[on_columns]).all(axis=1)]
    return pd.concat([data_childes, data_wikipedia_filtered]).drop_duplicates(subset=on_columns)


def mp_load_data(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, lang):
    # Load data
    nouns_data_childes = pd.read_csv(nouns_file_childes)
    nouns_data_wikipedia = pd.read_csv(nouns_file_wikipedia)
    verbs_data_childes = pd.read_csv(verbs_file_childes)
    verbs_data_wikipedia = pd.read_csv(verbs_file_wikipedia)

    if lang in ['fr', 'de']:
        # Handle noun and verb merging for French and German
        nouns_data = merge_and_remove_duplicates(nouns_data_childes, nouns_data_wikipedia, ['word_sing', 'word_plur', 'gender'])
        verbs_data = merge_and_remove_duplicates(verbs_data_childes, verbs_data_wikipedia, ['word_sing', 'word_plur', 'long_vp'])
    else:
        # Handle noun and verb merging for other languages (including English)
        nouns_data = merge_and_remove_duplicates(nouns_data_childes, nouns_data_wikipedia, ['word_sing', 'word_plur'])
        verbs_data = merge_and_remove_duplicates(verbs_data_childes, verbs_data_wikipedia, ['word_sing', 'word_plur', 'long_vp'])

    return nouns_data, verbs_data

def mp_load_partial_data(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, df):
    # Load data
    nouns_data_childes = pd.read_csv(nouns_file_childes)
    nouns_data_wikipedia = pd.read_csv(nouns_file_wikipedia)
    verbs_data_childes = pd.read_csv(verbs_file_childes)
    verbs_data_wikipedia = pd.read_csv(verbs_file_wikipedia)

    if df == 'wiki':
        if 'gender' in nouns_data_wikipedia.columns:
            nouns_data = nouns_data_wikipedia[['word_sing', 'word_plur', 'gender']]
        else:
            nouns_data = nouns_data_wikipedia[['word_sing', 'word_plur']]
        
        verbs_data = verbs_data_wikipedia[['word_sing', 'word_plur', 'long_vp']]
    elif df == 'childes':
        if 'gender' in nouns_data_childes.columns:
            nouns_data = nouns_data_childes[['word_sing', 'word_plur', 'gender']]
        else:
            nouns_data = nouns_data_childes[['word_sing', 'word_plur']] 
        verbs_data = verbs_data_childes[['word_sing', 'word_plur', 'long_vp']]
    else:
        raise ValueError("Invalid df option. Must be 'wiki' or 'childes'.")

    return nouns_data, verbs_data



def get_objects_by_source(obj_file_both, lang,df):

    objects_df = pd.read_csv(obj_file_both)
    
   
    if lang == 'fr' or lang == 'en':
        columns = ['word_sing', 'word_plur', 'gender']
    else:
        columns = ['word_sing', 'word_plur', 'gender','dative_sing', 'dative_plur']

    if df == 'wiki':
        return objects_df[objects_df['df'] == 'wiki'][columns]
    elif df == 'childes':
        return objects_df[objects_df['df'] == 'childes'][columns]



