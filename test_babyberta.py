from pyexpat import model
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoTokenizer
from minicons import scorer
import os
import numpy as np
from utils.variables import *
from transformers import AutoModel, AutoTokenizer, AutoConfig



def clams_score(model_path, eval_path):

    paradigm_accuracies = {}
    lm = scorer.MaskedLMScorer(model_path, 'cpu')
    
    for paradigm in os.listdir(eval_path):
        with open(os.path.join(eval_path, paradigm)) as f:
            sentences = []
            for line in f:
                sentences.append(line.strip()) 
            correct_count = 0
            total_count = 0
            for i in range(0, len(sentences), 2):
                grammatical = sentences[i].lower().replace('  ', ' ')
                ungrammatical = sentences[i + 1].lower().replace('  ', ' ')
                stimuli = [grammatical, ungrammatical]
                
                scores = lm.sequence_score(stimuli, reduction = lambda x: x.sum(0).item(), PLL_metric='original')
                if scores[0] > scores[1]:
                    correct_count += 1
                total_count += 1

            paradigm_accuracies[paradigm] = correct_count / total_count

    avg_accuracy = np.mean(list(paradigm_accuracies.values()))

    return avg_accuracy, paradigm_accuracies




def evaluate_models(models,eval_path):
    paradigm_names = []
    avg_accuracies = []
    all_paradigm_accuracies = []

    for model_path in models:
        avg_accuracy, paradigm_accuracies = clams_score(model_path, eval_path)
        print('Model:', model_path)
        print('Avg Accuracy:', avg_accuracy)
        print('Paradigm Accuracies:', paradigm_accuracies)
        avg_accuracies.append(avg_accuracy)
        all_paradigm_accuracies.append(paradigm_accuracies)

    # Calculate mean and std deviation
    mean_accuracy = np.mean(avg_accuracies)
    std_accuracy = np.std(avg_accuracies)

    # Aggregate accuracies per paradigm
    paradigm_means = {}
    paradigm_stds = {}
    for paradigm in all_paradigm_accuracies[0].keys():
        scores = [acc[paradigm] for acc in all_paradigm_accuracies]
        paradigm_means[paradigm] = np.mean(scores)
        paradigm_stds[paradigm] = np.std(scores)
        paradigm_names.append(paradigm)

    return mean_accuracy, std_accuracy, paradigm_means, paradigm_stds, paradigm_names


#here you can change the model you want to evaluate among those saved in the BabyBERTa directory
model_path = os.path.join(BABYBERTA_DIR,"saved_models/BabyBERTa_Wikipedia-1")
clams_path = CLAMS_DIR + '/en_evalset_ok'

avg_zorro, std_zorro, _,_,_ = evaluate_models([model_path], ZORRO_DIR)
print('Mean accuracy Zorro: ', avg_zorro)

avg_clams, std_clams, _, _, _= evaluate_models([model_path], clams_path)
print('Mean accuracy CLAMS: ', avg_clams)

avg_blimp, std_blimp, _,_,_ = evaluate_models([model_path], BLIMP_DIR)
print('Mean accuracy Blimp: ', avg_blimp)

