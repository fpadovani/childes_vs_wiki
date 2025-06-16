import os
import argparse
from torch.nn import CrossEntropyLoss
import logging
from pathlib import Path
from wrapper import *
from create_dataset_splits import *
import torch.nn.functional as F
import math 
import random 
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from logging.handlers import RotatingFileHandler
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from custom_functions import *

import transformers
from transformers import (
    RobertaConfig, 
    RobertaForMaskedLM,
    GPT2LMHeadModel,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    TrainerCallback,
    EarlyStoppingCallback,
    RobertaTokenizerFast,
    AutoTokenizer,
    SchedulerType,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)
from torch.optim import AdamW
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from utils.utils import *



# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler with rotation
file_handler = RotatingFileHandler('training.log', maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.addHandler(file_handler)


# Configure basic logging for all processes (console output)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help='The integration to report the results and logs to (e.g., "wandb").',
    )

    parser.add_argument(
        '--wandb_project',
        type=str,
        required=True,
        help='Provide the name for the wandb project'
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Pretrained tokenizer name or path."
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate to use.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Vocabulary size of the tokenizer.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use.",
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before a backward pass.",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hugging Face Hub.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the final model.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gpt2"],  # Add more if needed
        required=True,
        help="Model type to use if training from scratch.",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust code from the Hub.",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Folder in which the dataset splits are stored.",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length for model input.",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The target language of the model.",
    )

    parser.add_argument(
        "--validation_type",
        type=str,
        choices=["validation", "validation_ctc"],
        required=True,
        help="Choose between 'validation' (for Wikipedia) or 'validation_ctc' (for CHILDES).",
    )

    parser.add_argument(
        "--order",
        type=str,
        required=True,
        help="Provide an order label for the dataset (e.g., wikipedia_fr).",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input training file.",
    )

    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )

    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    return parser.parse_args()


def main():
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    val_log_file = os.path.join(args.output_dir, "validation_batches.log")

    send_example_telemetry("run_clm_no_trainer", args)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    logger.info(f"Accelerator state: {accelerator.state}")
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                if args.language:
                    repo_name = args.output_dir.split("/")[-1] + '_' + args.language + str(args.vocab_size)
                else:
                    repo_name = args.output_dir.split("/")[-1]
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    dataset_path = os.path.join(args.dataset_folder, args.language, args.order)
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset folder '{dataset_path}' exists.")
        raw_datasets = load_existing_dataset(os.path.join(args.dataset_folder, args.language), args.order)
    else:
        raw_datasets = create_and_load_new_dataset(args.input_file,args.dataset_folder, args.order, args.language)

    # Determine the path for the shuffled dataset
    shuffle_path = os.path.join(dataset_path, 'epochs', str(args.num_shuffling))

    # Check if shuffled dataset exists
    if os.path.exists(shuffle_path):
        raw_dataset_final = load_existing_dataset(args.os.path.join(args.dataset_folder, args.language), shuffle_path)
    else:
        raw_dataset_final = handle_dataset_scrambling(
            raw_datasets, args.order, args.num_shuffling, 
            args.dataset_folder, args.scramble_unit, args.streaming, args.language
        )

    remove_columns = raw_datasets["train"].column_names


    if os.path.exists(args.tokenizer_name):
        print(f"Tokenizer folder '{args.tokenizer_name}' exists.")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.tokenizer_name,
            model_max_length=128,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
            additional_special_tokens=["<|endoftext|>"])
                
    else:
        tokenizer = train_unified_tokenizer(dataset_files=args.input_file, save_path=args.tokenizer_name)
    

    
            
    concat_all_sentences = True
    minimal_sentence_length = 0
    maximal_sentence_length = args.context_length

    with accelerator.main_process_first():
        tokenized_datasets = raw_dataset_final.map(
        tokenize_wrapper_mlm(
            tokenizer,
            concat_all_sentences=concat_all_sentences,
            minimal_sentence_length=minimal_sentence_length,
            maximal_sentence_length=maximal_sentence_length,
        ),
        batched=True,
        remove_columns= remove_columns
    )
        
    tokenized_datasets = tokenized_datasets.map(
    lambda example: add_attention_mask_and_labels(example, tokenizer.pad_token_id),
    batched=False)
    
    if args.order in ['wikipedia', 'wikipedia_fr', 'wikipedia_de']:
        train_dataset = tokenized_datasets['train']
        valid_dataset = tokenized_datasets['validation']
    
    else:
        train_dataset = tokenized_datasets["train"]
        valid_dataset = tokenized_datasets[args.validation_type]



    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, mlm = True)

    config = RobertaConfig(
        vocab_size=8192,                # Vocabulary size
        num_hidden_layers=8,            # Number of Transformer layers
        num_attention_heads=8,          # Number of attention heads
        hidden_size=256,                # Hidden size
        intermediate_size=2048,
        layer_norm_eps=1e-5,            # Layer normalization epsilon
        eos_token_id=4,                 # End-of-sequence token ID
        bos_token_id=3,                 # Beginning-of-sequence token ID
        pad_token_id=1,                 # Padding token ID
        tie_word_embeddings=False,      # Do not tie word embeddings
    )


    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.mask_token_id = tokenizer.mask_token_id
    
    model = RobertaForMaskedLM(config)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Custom RoBERTa model size: {model_size / 1e6:.2f}M parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    validation_logging_callback = ValidationLoggingCallback(val_log_file, tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=100000
)
    
    checkpoint_steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


    args = TrainingArguments(
        output_dir=args.output_dir,
        seed = args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_steps=2000,
        logging_steps=4000,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay, 
        save_steps=4000,
        learning_rate = args.learning_rate,
        report_to="wandb",
        run_name = args.output_dir.split('/')[-2]+ '_' + args.output_dir.split('/')[-1],
        use_mps_device=False,
        fp16=False,
        push_to_hub=True,
        load_best_model_at_end=True,
        hub_strategy="all_checkpoints",
        metric_for_best_model="eval_loss",
        hub_token=args.hub_token,
        max_steps=10000,
        logging_dir=os.path.join(args.output_dir, "logs")
        )
    
    trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    callbacks=[validation_logging_callback,CustomCheckpointCallback(args.output_dir, checkpoint_steps, push_to_hub=True, repo_id=repo_id, hub_token=args.hub_token)],
    data_collator=data_collator,
    optimizers=(optimizer,scheduler),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset

)
    
    trainer.train()

    trainer.save_model(args.output_dir)
    

if __name__ == "__main__":
    main()