# childes_vs_wiki

This is the official repository of the paper ["Child-Directed Language Does Not Consistently Boost Syntax Learning in Language Models"](https://arxiv.org/abs/2505.23689).

## 1. 🏋️‍♂️ Training

The models (3 seeds) trained on size-matched Wikipedia and CHILDES are available on [Hugging Face](https://huggingface.co/fpadovani).  
The specific checkpoint steps used for evaluation are listed in the paper.

### 📁 Dataset Structure

The datasets used for training are available under the `./corpora` folder, organized into **CHILDES** and **Wikipedia** directories.  
In the `datasets` folder, you will find the **train/validation splits** in Hugging Face format. The split named `random` refers to the CHILDES data.

- The **train/valid split** was created by selecting **10%** of the transcripts from the CHILDES dataset.
- All evaluation sentences found in the training set were removed to avoid duplicates.
- The final evaluation dataset amounts to **8%** of the total data.
- Once we have counted the total tokens for the three languages as the sum between training and validation, we contrained the creation of the Wikipedia dataset to that amount of tokens. And we generated the evaluation split by taking the 8% of the total data.


### ⚙️ Training Script

To train the GPT-2 models, use the following command:

```
python3 train/clm_trainer.py \
  --report_to "wandb" \
  --wandb_project "name_of_your_wandb_proj" \
  --tokenizer_name "tokenizers/tokenizer_fr_wiki" \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 0.0001 \
  --weight_decay 0.01 \
  --seed 42 \
  --vocab_size 8192 \
  --lr_scheduler_type linear \
  --num_warmup_steps 40000 \
  --gradient_accumulation_steps 2 \
  --push_to_hub \
  --output_dir models_trained/name_of_the_model \
  --model_type gpt2 \
  --trust_remote_code \
  --dataset_folder datasets \
  --context_length 128 \
  --language french \
  --validation_type validation_ctc (for childes) or validation (for wikipedia) \
  --order wikipedia_fr \
  --input_file corpora/french/wikipedia/wikipedia_final.txt
```

To train the RoBERTa models, use the following command:

```
python3 train/masked_roberta_new.py \
  --report_to "wandb" \
  --wandb_project "name_of_your_wandb_proj" \
  --tokenizer_name "tokenizers/tokenizer_fr_wiki" \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 0.0001 \
  --weight_decay 0.01 \
  --seed 42 \
  --vocab_size 8192 \
  --lr_scheduler_type linear \
  --num_warmup_steps 40000 \
  --gradient_accumulation_steps 2 \
  --push_to_hub \
  --output_dir models_trained/name_of_the_model \
  --model_type gpt2 \
  --trust_remote_code \
  --dataset_folder datasets \
  --context_length 128 \
  --language french \
  --validation_type validation_ctc (for childes) or validation (for wikipedia) \
  --order wikipedia_fr \
  --input_file corpora/french/wikipedia/wikipedia_final.txt
  ```

## 2. Evaluation of Existing Benchmarks
In the folder `evaluation/test_suites`, you will find the three minimal pairs benchmarks used to evaluate our models:

- *zorro*
- *blimp*
- *clams/de_evalset_ok*
- *clams/en_evalset_ok*
- *clams/fr_evalset_ok*

Filtered versions are also available (excluding sentences with unseen tokens during training).
However, the evaluation reported in the paper uses the full versions of these benchmarks.

-- Run the following script to assign and save probabilities for each sentence in the evaluation benchmarks:

```evaluation/scripts/save_validation_probabilities.py```

Results will be saved in the **evaluation_probabilities/** directory under each benchmark's name.

-- Use this script to compute accuracy and generate bar plots based on the saved probabilities:

```evaluation/scripts/compute_accuracy_plots.py```

-- Run this script to compute per-paradigm scores (for each benchmark) and save results as JSON files in **json_results_clm/**:

```scripts/evaluate_curves_seeds.py```

Then, plot learning curves using:

```scripts/plots_curves_seeds.py```

## 3. FIT-CLAMS generation
Before proceeding with the generation of the new minimal pairs, it is necessary to compuute the unigram and bigram frequency distributions of the training dataset used in this work using the script ```n_gram_calculations_simple.py```.

Once run, the resulting frequency distributions will be saved in the folder **n_grams_frequency**
Additionally, you can compute dependency-based unigram and bigram frequencies using the script ```n_gram_calculations_dep.py```.
This script also saves its outputs in the same folder: **n_grams_frequency**.


Once you have these files ready, you can proceed with the following steps in order:

1. Run the script ```fitclams_generation/generate_bins_CLAMS_all_langs.py```, which first computes the shared vocabulary between CHILDES and Wikipedia for the three target languages (English, French, and German). Then, it groups nouns and verbs into 10 bins based on their observed frequency in the training dataset.
As a result, you will obtain 10 .csv files stored under each of the following folder names:

- *fitclams_generation/extracted_words/de-en-fr/nouns_per_bin_childes*

- *fitclams_generation/extracted_words/de-en-fr/verbs_per_bin_childes*

- *fitclams_generation/extracted_words/de-en-fr/nouns_per_bin_wiki*

- *fitclams_generation/extracted_words/de-en-fr/verbs_per_bin_wiki*


2. Once the bins are generated, we manually create a file named *chosen_small.csv* inside each of the previously mentioned folders.
This file contains the selected lexical items (verbs and nouns) for that specific dataset.

Additionally, for each folder corresponding to a language, we create another file named *objects_both.csv*, which contains the extra nouns to be used in the relative clause paradigms.

3. After selecting the lexical items, you can run the script ```fitclams_generation/generate_fit_clams_items.py``` to generate the actual new minimal pairs.
These are stored in the following folder:

*evaluation/test_suites/fit_clams*

Inside each language subfolder, you will find the minimal pairs generated based on the frequency distributions from Wikipedia and CHILDES, respectively.

## 4. FIT-CLAMS Evaluation

To evaluate FIT-CLAMS, use the script ```evaluation/scripts/save_validation_probabilities_fitclams.py```. This script saves the model probabilities in the folder **evaluation_probabilities/FIT_CLAMS**. To compute accuracy scores and generate bar plots comparing different models, use ```evaluation/scripts/compute_accuracy_plots.py```.


## 5. Regression Analysis

The notebook ```regression_analysis.ipynb``` allows you to curate the dataset on which linear regressions are then performed for the three languages and the two datasets (CHILDES and Wikipedia).
In the **regression_analysis** folder, you can find the datasets, regression scripts, and the generated plots used for the analysis.

These two files are important to perform dependency parsing on both the training datasets and evaluation benchmarks minimal pairs:
- ```parsing_files/parsing_clams.py```
- ```parsing_files/parsing_datasets.py```

The *.conllu* files and the *.pickle* files are going to be saved in these folders:
1. parsed_datasets/clams
2. parsed_datasets/fit_clams
3. parsed_datasets/training_datasets







