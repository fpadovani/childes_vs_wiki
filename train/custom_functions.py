from transformers import TrainerCallback
import os
    

def log_validation_batch(eval_dataloader, tokenizer, device, log_file):
    """
    Logs validation batch content (decoded tokens) to a file.
    """
    with open(log_file, "a") as log:
        log.write("\n--- Validation Batch Start ---\n")
        
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Decode the input IDs back into tokens
            decoded_tokens = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            
            for i, tokens in enumerate(decoded_tokens):
                log.write(f"Sample {i}:\n{tokens}\n")
            
            log.write("--- Validation Batch End ---\n\n")
            break  # Log only the first batch for efficiency


class ValidationLoggingCallback(TrainerCallback):
    """
    Custom callback to log validation batches whenever validation is performed.
    """
    def __init__(self, log_file, tokenizer):
        self.log_file = log_file
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Triggered during evaluation to log validation batch content.
        """
        eval_dataloader = kwargs.get("eval_dataloader")
        if eval_dataloader:
            log_validation_batch(eval_dataloader, self.tokenizer, args.device, self.log_file)


    
class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, checkpoint_steps, push_to_hub=False, repo_id=None, hub_token=None):
        self.output_dir = output_dir
        self.checkpoint_steps = checkpoint_steps
        self.push_to_hub = push_to_hub
        self.repo_id = repo_id  # Make sure to pass repo_id here
        self.hub_token = hub_token  # Hub token for authentication

    def on_step_end(self, args, state, control, **kwargs):
        """Save model only at predefined steps and optionally push to Hugging Face Hub."""
        if state.global_step in self.checkpoint_steps:
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            
            # Save locally
            kwargs['model'].save_pretrained(checkpoint_path)
            kwargs['tokenizer'].save_pretrained(checkpoint_path)
            print(f"Checkpoint saved at step {state.global_step}")

            # Push to Hugging Face Hub if enabled
            if self.push_to_hub and self.repo_id:
                print(f"Pushing checkpoint-{state.global_step} to the hub...")
                kwargs['model'].push_to_hub(
                    repo_id=self.repo_id,
                    commit_message=f"Checkpoint-{state.global_step}",
                    token=self.hub_token  # Ensure you provide the hub token here
                )
                kwargs['tokenizer'].push_to_hub(
                    repo_id=self.repo_id,
                    commit_message=f"Checkpoint-{state.global_step}",
                    token=self.hub_token  # Ensure you provide the hub token here
                )
