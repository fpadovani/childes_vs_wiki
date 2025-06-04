import pandas as pd
import os
import itertools
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_augmentation import *
from utils.variables import *

LANG_PROPS = {
    'en': {
        'conjunction': 'and',
        'rel_pron': {'sing': 'that ', 'plur': 'that '},
        'rel_verbs': {
            'sing': ['likes', 'hates', 'loves', 'admires'],
            'plur': ['like', 'hate', 'love', 'admire']
        },
        'articles': lambda get_articles: {
            'subject': get_articles['subject'],
            'subject_plur': get_articles['subject_plur'],
            'object': get_articles['object'],
            'object_plur': get_articles['object_plur']
        },
        'comma': ''
    },
    'fr': {
        'conjunction': 'et',
        'rel_pron': {'sing': 'qui ', 'plur': 'qui '},
        'rel_verbs': {'sing': ['aime'], 'plur': ['aiment']},
        'articles': lambda get_articles: {
            'subject': get_articles['subject'],
            'subject_plur': get_articles['subject_plur'],
            'object': get_articles['object'],
            'object_plur': get_articles['object_plur']
        },
        'comma': ''
    },
    'de': {
        'conjunction': 'und',
        'rel_pron': {'plur': 'die '},  # singular added dynamically
        'rel_verbs': {'sing': ['mag', 'vermeidet'], 'plur': ['mögen', 'vermeiden']},
        'articles': lambda get_articles: {
            'subject': get_articles['subject'],
            'subject_plur': get_articles['subject_plur'],
            'object': get_articles['object'],
            'object_plur': get_articles['object_plur'],
            'relative_pron_sing': get_articles['relative_pron_sing_nominative']
        },
        'comma': ' ,'
    }
}


    

def get_lang_config(lang):
    props = LANG_PROPS[lang]
    
    # Use empty string as placeholder for dynamically generated pronouns
    rel_pron_sing = props['rel_pron'].get('sing', '')

    return (
        rel_pron_sing,
        props['rel_pron']['plur'],
        props['rel_verbs']['sing'],
        props['rel_verbs']['plur'],
        props['comma'],
        props['conjunction']
    )


def get_articles_by_lang(lang, noun, gender):
    base_articles = get_articles(lang)
    article_funcs = LANG_PROPS[lang]['articles'](base_articles)

    def resolve(article, word):
        return article(word, gender) if callable(article) else article

    result = {
        'subject': resolve(article_funcs['subject'], noun),
        'subject_plur': article_funcs['subject_plur'],
        'object': resolve(article_funcs['object'], noun),
        'object_plur': article_funcs['object_plur'],
        'relative_pron_sing': ''
    }

    if lang == 'de' and 'relative_pron_sing' in article_funcs:
        result['relative_pron_sing'] = article_funcs['relative_pron_sing'](noun, gender)

    return result

def get_relative_prons_and_verbs(lang):
    props = LANG_PROPS[lang]
    rel_pron = props['rel_pron']
    
    # Use .get to avoid KeyError
    rel_pron_sing = rel_pron.get('sing', '')  # fallback to empty string
    rel_pron_plur = rel_pron['plur']
    
    return (
        rel_pron_sing,
        rel_pron_plur,
        props['rel_verbs']['sing'],
        props['rel_verbs']['plur']
    )


def get_obj_rel_pronoun_de(gender: str, number: str) -> str:
    """Returns the correct accusative relative pronoun for German based on object gender and number."""
    if number == 'sing':
        if gender == 'Masc':
            return 'den'
        elif gender == 'Fem':
            return 'die'
        elif gender == 'Neut':
            return 'das'
    elif number == 'plur':
        return 'die'
    return ''

def mp_subj_rel(
    nouns_file_childes,
    nouns_file_wikipedia,
    verbs_file_childes,
    verbs_file_wikipedia,
    obj_file_both,
    output_file,
    opt,
    lang,
):
    # Load lexical data
    nouns_data, verbs_data = load_data_by_option(
        nouns_file_childes,
        nouns_file_wikipedia,
        verbs_file_childes,
        verbs_file_wikipedia,
        opt,
        lang,
    )
    obj_data = get_objects_by_source(obj_file_both, lang, df=opt)

    # Language-specific props
    lang_props = LANG_PROPS[lang]
    rel_verbs_sing = lang_props["rel_verbs"]["sing"]
    rel_verbs_plur = lang_props["rel_verbs"]["plur"]
    rel_pron_plur = lang_props["rel_pron"]["plur"]
    rel_pron_static_sing = lang_props["rel_pron"].get("sing", "")
    comma = lang_props["comma"]

    minimal_pairs = []

    for _, noun_row in nouns_data.iterrows():
        subj_sing = noun_row["word_sing"]
        subj_plur = noun_row["word_plur"]
        gender = noun_row["gender"]

        # Fetch subject articles and relative pronoun
        noun_articles = get_articles_by_lang(lang, subj_sing, gender)
        art_subj_sing = noun_articles["subject"]
        art_subj_plur = noun_articles["subject_plur"]
        rel_pron_func = noun_articles.get("relative_pron_sing", None)
        rel_pron_sing = rel_pron_func(subj_sing, gender) if callable(rel_pron_func) else rel_pron_static_sing

        for _, obj_row in obj_data.iterrows():
            obj_sing = obj_row["word_sing"]
            obj_plur = obj_row["word_plur"]
            obj_gender = obj_row["gender"]

            # Fetch object articles
            obj_articles = get_articles_by_lang(lang, obj_sing, obj_gender)
            art_obj_sing = obj_articles["object"]
            art_obj_plur = obj_articles["object_plur"]

            for _, verb_row in verbs_data.iterrows():
                verb_sing = verb_row["word_sing"]
                verb_plur = verb_row["word_plur"]

                for i in range(len(rel_verbs_sing)):
                    rel_verb_sing = rel_verbs_sing[i]
                    rel_verb_plur = rel_verbs_plur[i]

                    if lang == "de":
                        # German-specific structure (dynamic singular rel pron)
                        minimal_pairs.extend([
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{art_obj_sing}{obj_sing} {rel_verb_sing}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{art_obj_sing}{obj_sing} {rel_verb_plur}{comma} {verb_sing}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{art_obj_plur}{obj_plur} {rel_verb_sing}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{art_obj_plur}{obj_plur} {rel_verb_plur}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{art_obj_sing}{obj_sing} {rel_verb_plur}{comma} {verb_plur}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{art_obj_sing}{obj_sing} {rel_verb_sing}{comma} {verb_plur}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{art_obj_plur}{obj_plur} {rel_verb_plur}{comma} {verb_plur}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{art_obj_plur}{obj_plur} {rel_verb_sing}{comma} {verb_plur}"],
                        ])
                    else:
                        # English/French/Other
                        minimal_pairs.extend([
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{rel_verb_sing} {art_obj_sing}{obj_sing}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{rel_verb_plur} {art_obj_sing}{obj_sing}{comma} {verb_sing}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{rel_verb_sing} {art_obj_plur}{obj_plur}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{rel_verb_plur} {art_obj_plur}{obj_plur}{comma} {verb_sing}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{rel_verb_plur} {art_obj_sing}{obj_sing}{comma} {verb_plur}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{rel_verb_sing} {art_obj_sing}{obj_sing}{comma} {verb_plur}"],
                            [f"{art_subj_plur}{subj_plur}{comma} {rel_pron_plur}{rel_verb_plur} {art_obj_plur}{obj_plur}{comma} {verb_plur}"],
                            [f"{art_subj_sing}{subj_sing}{comma} {rel_pron_sing}{rel_verb_sing} {art_obj_plur}{obj_plur}{comma} {verb_plur}"],
                        ])

    pd.DataFrame(minimal_pairs).to_csv(output_file, index=False, header=False)


def mp_simple_agrmt(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, output_file, opt, lang):
    nouns_data, verbs_data = load_data_by_option(
        nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, opt, lang)
    minimal_pairs = []

    for _, noun_row in nouns_data.iterrows():
        noun_sing, noun_plur, gender = noun_row['word_sing'], noun_row['word_plur'], noun_row['gender']
        articles = get_articles_by_lang(lang, noun_sing, gender)
        art_sing, art_plur = articles['subject'], articles['subject_plur']

        for _, verb_row in verbs_data.iterrows():
            verb_sing, verb_plur = verb_row['word_sing'], verb_row['word_plur']

            minimal_pairs.append([f"{art_sing}{noun_sing} {verb_sing}"])
            minimal_pairs.append([f"{art_plur}{noun_plur} {verb_sing}"])
            minimal_pairs.append([f"{art_plur}{noun_plur} {verb_plur}"])
            minimal_pairs.append([f"{art_sing}{noun_sing} {verb_plur}"])

    pd.DataFrame(minimal_pairs).to_csv(output_file, index=False, header=False)



def mp_vp_coord(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, output_file, opt, lang):
    nouns_data, verbs_data = load_data_by_option(
        nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, opt, lang)
    minimal_pairs = []
    conjunction = LANG_PROPS[lang]['conjunction']

    for _, noun_row in nouns_data.iterrows():
        noun_sing, noun_plur, gender = noun_row['word_sing'], noun_row['word_plur'], noun_row['gender']
        articles = get_articles_by_lang(lang, noun_sing, gender)
        art_sing, art_plur = articles['subject'], articles['subject_plur']

        verb_pairs = list(itertools.combinations(verbs_data.iterrows(), 2))

        for (verb1_row, verb2_row) in verb_pairs:
            verb1_sing, verb1_plur = verb1_row[1]['word_sing'], verb1_row[1]['word_plur']
            verb2_sing, verb2_plur = verb2_row[1]['word_sing'], verb2_row[1]['word_plur']

            minimal_pairs.extend([
                [f"{art_sing}{noun_sing} {verb1_sing} {conjunction} {verb2_sing}"],
                [f"{art_plur}{noun_plur} {verb1_plur} {conjunction} {verb2_sing}"],
                [f"{art_plur}{noun_plur} {verb1_plur} {conjunction} {verb2_plur}"],
                [f"{art_sing}{noun_sing} {verb1_sing} {conjunction} {verb2_plur}"],
            ])

    pd.DataFrame(minimal_pairs).to_csv(output_file, index=False, header=False)



def mp_long_vp_coord(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, output_file, opt, lang):
    nouns_data, verbs_data = load_data_by_option(
        nouns_file_childes, nouns_file_wikipedia,
        verbs_file_childes, verbs_file_wikipedia,
        opt, lang
    )

    conjunction = LANG_PROPS[lang]['conjunction']
    minimal_pairs = []

    for _, noun_row in nouns_data.iterrows():
        noun_sing, noun_plur, gender = noun_row['word_sing'], noun_row['word_plur'], noun_row['gender']
        art = get_articles_by_lang(lang, noun_sing, gender)

        verb_pairs = list(itertools.combinations(verbs_data.iterrows(), 2))
        for (verb1_row, verb2_row) in verb_pairs:
            verb1_sing, verb1_plur, attr1 = verb1_row[1]['word_sing'], verb1_row[1]['word_plur'], verb1_row[1]['long_vp']
            verb2_sing, verb2_plur, attr2 = verb2_row[1]['word_sing'], verb2_row[1]['word_plur'], verb2_row[1]['long_vp']

            minimal_pairs.extend([
                [f"{art['subject']}{noun_sing} {verb1_sing} {attr1} {conjunction} {verb2_sing} {attr2}"],
                [f"{art['subject_plur']}{noun_plur} {verb1_plur} {attr1} {conjunction} {verb2_sing} {attr2}"],
                [f"{art['subject_plur']}{noun_plur} {verb1_plur} {attr1} {conjunction} {verb2_plur} {attr2}"],
                [f"{art['subject']}{noun_sing} {verb1_sing} {attr1} {conjunction} {verb2_plur} {attr2}"]
            ])

    pd.DataFrame(minimal_pairs).to_csv(output_file, index=False, header=False)




def mp_obj_rel_across(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, obj_file_both, output_file, opt, lang):

    nouns_data, verbs_data = load_data_by_option(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, opt, lang)
    obj_data = get_objects_by_source(obj_file_both,lang, df=opt)

    minimal_pairs = []
    articles = get_articles(lang)
    
    lang_props = LANG_PROPS[lang]
    rel_verbs_sing = lang_props['rel_verbs']['sing']
    rel_verbs_plur = lang_props['rel_verbs']['plur']
    if lang != 'de':
        rel_pron_sing = lang_props['rel_pron']['sing']
    rel_pron_plur = lang_props['rel_pron']['plur']
    comma = lang_props['comma']

    article_access = LANG_PROPS[lang]['articles'](get_articles(lang))


    for _, noun_row in nouns_data.iterrows():
        subj_sing, subj_plur, gender = noun_row['word_sing'], noun_row['word_plur'], noun_row.get('gender')

        art_s_func = article_access['subject']
        art_sing = art_s_func(subj_sing, gender) if callable(art_s_func) else art_s_func
        art_plur = article_access['subject_plur']

        if lang == 'de':
            rel_pron_sing = articles['relative_pron_sing_accusative'](subj_sing, gender)
        
        if lang == 'fr':
            rel_pron_sing = 'que '
            rel_pron_plur = 'que '

        for _, obj_row in obj_data.iterrows():
            obj_sing, obj_plur = obj_row['word_sing'], obj_row['word_plur']
            obj_gender = obj_row.get('gender') or 'Masc'

            art_o_func = article_access['object']
            art_o_sing = art_o_func(obj_sing, obj_gender) if callable(art_o_func) else art_o_func
            art_o_plur = article_access['object_plur']

            # Handle verbs and generate minimal pairs
            for _, verb_row in verbs_data.iterrows():
                verb_sing, verb_plur = verb_row['word_sing'], verb_row['word_plur']

                for i in range(len(rel_verbs_sing)):
                    minimal_pairs.append([art_sing + subj_sing + f"{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verbs_sing[i]}{comma} {verb_sing}"])
                    minimal_pairs.append([art_plur + subj_plur + f"{comma} {rel_pron_plur}{art_o_sing}{obj_sing} {rel_verbs_sing[i]}{comma} {verb_sing}"])
                    minimal_pairs.append([art_sing + subj_sing + f"{comma} {rel_pron_sing}{art_o_plur}{obj_plur} {rel_verbs_plur[i]}{comma} {verb_sing}"])
                    minimal_pairs.append([art_plur + subj_plur + f"{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verbs_plur[i]}{comma} {verb_sing}"])

                    minimal_pairs.append([art_plur + subj_plur + f"{comma} {rel_pron_plur}{art_o_sing}{obj_sing} {rel_verbs_sing[i]}{comma} {verb_plur}"])
                    minimal_pairs.append([art_sing + subj_sing + f"{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verbs_sing[i]}{comma} {verb_plur}"])
                    minimal_pairs.append([art_plur + subj_plur + f"{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verbs_plur[i]}{comma} {verb_plur}"])
                    minimal_pairs.append([art_sing + subj_sing + f"{comma} {rel_pron_sing}{art_o_plur}{obj_plur} {rel_verbs_plur[i]}{comma} {verb_plur}"])


    minimal_pairs_df = pd.DataFrame(minimal_pairs)
    minimal_pairs_df.to_csv(output_file, index=False,header=False)



def mp_obj_rel_within(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia,
                      obj_file_both, output_file, opt, lang):

    nouns_data, verbs_data = load_data_by_option(nouns_file_childes, nouns_file_wikipedia,
                                                 verbs_file_childes, verbs_file_wikipedia, opt, lang)
    obj_data = get_objects_by_source(obj_file_both, lang, df=opt)
    articles = get_articles(lang)

    rel_pron_sing, rel_pron_plur, rel_verbs_sing, rel_verbs_plur, comma, conjunction = get_lang_config(lang)

    def get_article(article_type, word, gender=None):
        """Handles callable vs static article resolution."""
        article = articles[article_type]
        return article(word, gender) if callable(article) else article

    minimal_pairs = []

    for _, noun_row in nouns_data.iterrows():
        subj_sing, subj_plur = noun_row['word_sing'], noun_row['word_plur']
        gender = noun_row.get('gender')

        art_sing = get_article('subject', subj_sing, gender)
        art_plur = articles['subject_plur']

        if lang == 'de':
            rel_pron_sing = articles['relative_pron_sing_accusative'](subj_sing, gender)

        for _, obj_row in obj_data.iterrows():
            obj_sing, obj_plur = obj_row['word_sing'], obj_row['word_plur']
            obj_gender = obj_row.get('gender')

            art_o_sing = get_article('object', obj_sing, obj_gender)
            art_o_plur = articles['object_plur']

            for _, verb_row in verbs_data.iterrows():
                verb_sing, verb_plur = verb_row['word_sing'], verb_row['word_plur']

                for i in range(len(rel_verbs_sing)):
                    rel_verb_sing = rel_verbs_sing[i]
                    rel_verb_plur = rel_verbs_plur[i]

                    minimal_pairs.extend([
                        [f"{art_sing}{subj_sing}{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verb_sing}{comma} {verb_sing}"],
                        [f"{art_sing}{subj_sing}{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verb_sing}{comma} {verb_sing}"],
                        [f"{art_sing}{subj_sing}{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verb_plur}{comma} {verb_sing}"],
                        [f"{art_sing}{subj_sing}{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verb_plur}{comma} {verb_sing}"],
                        [f"{art_plur}{subj_plur}{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verb_sing}{comma} {verb_plur}"],
                        [f"{art_plur}{subj_plur}{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verb_sing}{comma} {verb_plur}"],
                        [f"{art_plur}{subj_plur}{comma} {rel_pron_plur}{art_o_plur}{obj_plur} {rel_verb_plur}{comma} {verb_plur}"],
                        [f"{art_plur}{subj_plur}{comma} {rel_pron_sing}{art_o_sing}{obj_sing} {rel_verb_plur}{comma} {verb_plur}"]
                    ])

    pd.DataFrame(minimal_pairs).to_csv(output_file, index=False, header=False)




def mp_prep_anim(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, obj_file_both, output_file, opt, lang):

    nouns_data, verbs_data = load_data_by_option(nouns_file_childes, nouns_file_wikipedia, verbs_file_childes, verbs_file_wikipedia, opt, lang)
    obj_data = get_objects_by_source(obj_file_both,lang, df=opt)

    minimal_pairs = []
    
    articles = get_articles(lang)
    if lang == 'fr':
        #devant and derrière don't require the genitive 
        prep_ = ['devant', 'derrière', 'en face', 'à côté', 'près']
    elif lang == 'de':
        prep_ = ['vor', 'hinter', 'neben', 'in der nähe von', 'gegenüber'] #after the vor and the in der nähe von you need the dative
    elif lang == 'en':
        prep_ = ['next to', 'behind', 'in front of', 'near', 'to the side of', 'across from']
    else:
        raise ValueError(f"Unsupported language: {lang}")

    # Loop through noun data and generate minimal pairs
    for _, noun_row in nouns_data.iterrows():
        subj_sing, subj_plur = noun_row['word_sing'], noun_row['word_plur']
        gender = noun_row.get('gender', None) 

        # For English, articles are directly strings, no need to call them
        if lang == 'en':
            art_sing = articles['subject']  
            art_plur = articles['subject_plur']
            genitive_sing = articles['genitive_sing']
            genitive_plur = articles['genitive_plur']
        elif lang == 'de':
            art_sing = articles['subject'](subj_sing, gender)
            art_plur = articles['subject_plur']
            genitive_sing = articles['genitive_sing']
            genitive_plur = articles['genitive_plur']
        else:
            art_sing = articles['subject'](subj_sing, gender)
            art_plur = articles['subject_plur']

        for _, obj_row in obj_data.iterrows():
            if lang == 'fr' or lang == 'en':
                obj_sing, obj_plur = obj_row['word_sing'], obj_row['word_plur']
            elif lang == 'de':
                obj_sing, obj_plur, dative_sing, dative_plur = obj_row['word_sing'], obj_row['word_plur'], obj_row['dative_sing'], obj_row['dative_plur']


            for _, verb_row in verbs_data.iterrows():
                verb_sing, verb_plur = verb_row['word_sing'], verb_row['word_plur']

                for prep in prep_:
                    if lang == 'fr':
                        gender_o = obj_row['gender']
                        art_o_sing, art_o_plur = articles['object'](obj_sing, gender_o), articles['object_plur']
                        # Determine if this preposition requires genitive
                        if prep in ['devant', 'derrière']:
                            gen_sing = art_o_sing
                            gen_plur = art_o_plur
                        else:
                            genitive_sing = articles['genitive_sing'](obj_sing,gender_o)
                            genitive_plur = articles['genitive_plur']
                            gen_sing = genitive_sing
                            gen_plur = genitive_plur

                    elif lang == 'de':
                        if prep in ['vor', 'in der nähe von']:
                            gen_sing = dative_sing
                            gen_plur = dative_plur
                            obj_sing = ''
                            obj_plur = ''
                    else:
                        gen_sing = genitive_sing
                        gen_plur = genitive_plur

                    minimal_pairs.append([art_sing + subj_sing + f" {prep} {gen_sing}{obj_sing} {verb_sing}"])
                    minimal_pairs.append([art_plur + subj_plur + f" {prep} {gen_sing}{obj_sing} {verb_sing}"])
                    minimal_pairs.append([art_sing + subj_sing + f" {prep} {gen_plur}{obj_plur} {verb_sing}"])
                    minimal_pairs.append([art_plur + subj_plur + f" {prep} {gen_plur}{obj_plur} {verb_sing}"])

                    minimal_pairs.append([art_plur + subj_plur + f" {prep} {gen_sing}{obj_sing} {verb_plur}"])
                    minimal_pairs.append([art_sing + subj_sing + f" {prep} {gen_sing}{obj_sing} {verb_plur}"])
                    minimal_pairs.append([art_plur + subj_plur + f" {prep} {gen_plur}{obj_plur} {verb_plur}"])
                    minimal_pairs.append([art_sing + subj_sing + f" {prep} {gen_plur}{obj_plur} {verb_plur}"])

    minimal_pairs_df = pd.DataFrame(minimal_pairs)
    minimal_pairs_df.to_csv(output_file, index=False, header=False)



LANGUAGE_CONFIG = {
    'fr': {
        'output_directories': {
            'noun_per_bin_childes': 'fr/nouns_per_bin_childes',
            'verb_per_bin_childes': 'fr/verbs_per_bin_childes',
            'noun_per_bin_wiki': 'fr/nouns_per_bin_wiki',
            'verb_per_bin_wiki': 'fr/verbs_per_bin_wiki',
        },
        'objects_file': "fitclams_generation/extracted_words/fr/objects_both.csv",
    },
    'en': {
        'output_directories': {
            'noun_per_bin_childes': 'en/nouns_per_bin_childes',
            'verb_per_bin_childes': 'en/verbs_per_bin_childes',
            'noun_per_bin_wiki': 'en/nouns_per_bin_wiki',
            'verb_per_bin_wiki': 'en/verbs_per_bin_wiki',
        },
        'objects_file': "fitclams_generation/extracted_words/en/objects_both.csv",
    },
    'de': {
        'output_directories': {
            'noun_per_bin_childes': 'de/nouns_per_bin_childes',
            'verb_per_bin_childes': 'de/verbs_per_bin_childes',
            'noun_per_bin_wiki': 'de/nouns_per_bin_wiki',
            'verb_per_bin_wiki': 'de/verbs_per_bin_wiki',
        },
        'objects_file': "fitclams_generation/extracted_words/de/objects_both.csv",
    }

}

def get_language_choice():
    print("Available languages: 'fr', 'en', 'de'")
    language_choice = input("Please select a language to process: ").strip().lower()
    if language_choice not in LANGUAGE_CONFIG:
        print("Invalid choice. Please select from the available languages.")
        return get_language_choice()
    return language_choice

def main():
    lang = get_language_choice()
    config = LANGUAGE_CONFIG[lang]

    noun_childes_path = os.path.join(EXTRACTED_WORDS, config['output_directories']['noun_per_bin_childes'], 'chosen_small.csv')
    verb_childes_path = os.path.join(EXTRACTED_WORDS, config['output_directories']['verb_per_bin_childes'], 'chosen_small.csv')
    noun_wiki_path = os.path.join(EXTRACTED_WORDS, config['output_directories']['noun_per_bin_wiki'], 'chosen_small.csv')
    verb_wiki_path = os.path.join(EXTRACTED_WORDS, config['output_directories']['verb_per_bin_wiki'], 'chosen_small.csv')
    objects_path = os.path.join(BASE_DIR, config['objects_file'])

    options = ['wiki', 'childes']
    for opt in options:
        output_dir = os.path.join(BASE_DIR, f"fitclams_generation/extracted_minimal_pairs_new_1/{lang}/{opt}")
        os.makedirs(output_dir, exist_ok=True)

        file_map = {
            "simple_agrmt": "simple_agrmt.csv",
            "vp_coord": "vp_coord.csv",
            "long_vp_coord": "long_vp_coord.csv",
            "subj_rel": "subj_rel.csv",
            "obj_rel_within": "obj_rel_within_anim.csv",
            "obj_rel_across": "obj_rel_across_anim.csv",
            "prep_anim": "prep_anim.csv"
        }

        paths = {key: os.path.join(output_dir, filename) for key, filename in file_map.items()}

        mp_simple_agrmt(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, paths["simple_agrmt"], opt, lang)
        mp_vp_coord(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, paths["vp_coord"], opt, lang)
        mp_long_vp_coord(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, paths["long_vp_coord"], opt, lang)
        mp_subj_rel(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, objects_path, paths["subj_rel"], opt, lang)
        mp_obj_rel_within(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, objects_path, paths["obj_rel_within"], opt, lang)
        mp_obj_rel_across(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, objects_path, paths["obj_rel_across"], opt, lang)
        mp_prep_anim(noun_childes_path, noun_wiki_path, verb_childes_path, verb_wiki_path, objects_path, paths["prep_anim"], opt, lang)

if __name__ == "__main__":
    main()
