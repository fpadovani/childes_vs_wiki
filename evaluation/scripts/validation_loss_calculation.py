import os
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from tqdm import tqdm
import json 
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.variables import TRAINED_MODELS, VALIDATION_LOSS_PLOTS


colors = ['#5A2D74', '#003B5C', '#50C878']

language_mapping = {
    'en': 'English',
    'de': 'German',
    'fr': 'French'
}


# Function to extract eval_loss and step values
def extract_eval_losses(checkpoint_path):
    eval_losses = []
    steps = []

    # Open the trainer_state.json file
    trainer_state_path = os.path.join(checkpoint_path,'checkpoint-100000', "trainer_state.json")
    
    # Check if the trainer_state.json file exists
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            
            # Check if the 'log_history' field exists in the trainer_state
            if 'log_history' in trainer_state:
                for entry in trainer_state['log_history']:
                    if 'eval_loss' in entry:
                        eval_losses.append(entry['eval_loss'])
                        steps.append(entry['step'])
    
    return steps, eval_losses


# Function to create and save the plot for all the models
def plot_eval_loss_for_models(models, output_dir, language, model_type, dataset_type):
    plt.figure(figsize=(10, 6))
    plt.title(f"Validation Loss for {model_type.upper()} in {language_mapping.get(language, language)} on {dataset_type.capitalize()}", fontsize=15)
    plt.xlabel('Steps', fontsize=13)
    plt.ylabel('Eval Loss', fontsize=13)

    # Initialize variables for dynamic y-axis limits
    min_loss = float('inf')
    max_loss = float('-inf')

    # Iterate over each model and plot the eval_loss
    for i, model in enumerate(models):
        steps, eval_losses = extract_eval_losses(model)
        if steps and eval_losses:
            label = os.path.basename(model).replace("_new", "")
            plt.plot(steps, eval_losses, label=label, color=colors[i % len(colors)], linestyle='-', linewidth=2) 
            
            # Update min and max loss for dynamic y-axis range
            min_loss = min(min_loss, min(eval_losses))
            max_loss = max(max_loss, max(eval_losses))
    
    # Set the y-axis range dynamically based on the losses
    plt.ylim(min_loss - 0.2, max_loss + 0.2) 
    
    plt.xticks(range(0, max(steps) + 1, 8000), fontsize=10)  # X-axis ticks at every 8000 steps
    plt.yticks(np.arange(min_loss - 0.5, max_loss + 0.5, 0.5), fontsize=10)  # Y-axis ticks with a step of 1
    
    plt.grid(axis='x', linestyle='--', linewidth=0.7, color='gray')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')

    if len(models) > 0:
        plt.legend(loc='best', fontsize=15)
    
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{language}_{model_type}_{dataset_type}_validation_loss.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()



# Define the configuration for the models and languages
languages = ['en', 'fr', 'de']
model_types = ['clm', 'mlm']
seeds = ['13', '30', '42']
dataset_types = ['child', 'wiki']



# Iterate over each language, model type, and dataset type
for language in languages:
    for model_type in model_types:
        for dataset_type in dataset_types:
            models = []
            
            # Create the folder path for the current language, model type, and dataset type
            for seed in seeds:
                checkpoint_dir = f"{TRAINED_MODELS}/{language}_{dataset_type}_{model_type}_{seed}_new"
                models.append(checkpoint_dir)

            # Output directory for this language and model type
            output_dir = os.path.join(VALIDATION_LOSS_PLOTS, language, model_type, dataset_type)
            
            # Plot and save the validation loss for the current configuration
            plot_eval_loss_for_models(models, output_dir, language, model_type, dataset_type)
