"""QLoRA fine-tuning script for Qwen model."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B"
    max_seq_length: int = 256
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training settings
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    
    # Paths
    data_path: Path = Path("data/dataset.jsonl")
    output_dir: Path = Path("checkpoints")

def load_jsonl(file_path: Path) -> list:
    """Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of samples
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def prepare_model(config: TrainingConfig):
    """Prepare model for training.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    
    # Quantization config
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        # CPU 설정
        bnb_config = None
        print("Warning: Running on CPU. This will be slow and might run out of memory.")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config if use_cuda else None,
        device_map="auto" if use_cuda else "cpu",
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    return model, tokenizer, lora_config

def main():
    """Main training script."""
    setup_logging()
    config = TrainingConfig()
    
    # Load model and tokenizer
    model, tokenizer, lora_config = prepare_model(config)
    
    # Load dataset
    dataset = load_dataset(
        'json',
        data_files={'train': str(config.data_path)},
        split='train'
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        max_seq_length=config.max_seq_length,
        packing=True,
    )
    
    # Train
    trainer.train()
    
    # Save trained model
    trainer.save_model()
    logging.info(f"Training completed. Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
