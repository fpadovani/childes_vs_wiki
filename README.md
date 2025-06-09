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

- zorro
- blimp
- clams/de_evalset_ok
- clams/en_evalset_ok
- clams/fr_evalset_ok

Filtered versions are also available (excluding sentences with unseen tokens during training).
However, the evaluation reported in the paper uses the full versions of these benchmarks.

-- Run the following script to assign and save probabilities for each sentence in the evaluation benchmarks:

```evaluation/scripts/save_validation_probabilities.py```

Results will be saved in the `evaluation_probabilities/` directory under each benchmark's name.

-- Use this script to compute accuracy and generate bar plots based on the saved probabilities:

```evaluation/scripts/compute_accuracy_plots.py```

-- Run this script to compute per-paradigm scores (for each benchmark) and save results as JSON files in `json_results_clm/`:

```scripts/evaluate_curves_seeds.py```

Then, plot learning curves using:

```scripts/plots_curves_seeds.py```


