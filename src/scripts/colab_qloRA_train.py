#!/usr/bin/env python3
"""
Colab-ready QLoRA training script for Llama 3.2 3B (or other compatible causal models).
- Expects training data in JSONL with records containing either:
  {"instruction":..., "input":..., "output":...}
  or
  {"content": ...} (simple single-field will be used as both instruction+input->output style)
- Uses bitsandbytes 4-bit load, PEFT/LoRA, and Transformers Trainer for simplicity.

Usage (in Colab):
  python colab_qloRA_train.py --data_file /content/qwen/train.jsonl --output_dir /content/qwen_out --model_name_or_path <HF_MODEL>

Note: Run in Colab with GPU (A100/T4/V100) and install packages first (see README_COLAB.md).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            items.append(data)
    return items


def build_text_from_record(rec: Dict) -> str:
    # Prefer explicit instruction/input/output structure
    if 'instruction' in rec and 'output' in rec:
        instr = rec.get('instruction', '').strip()
        inp = rec.get('input', '').strip()
        out = rec.get('output', '').strip()
        if inp:
            prompt = f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        else:
            prompt = f"Instruction: {instr}\nOutput: {out}"
        return prompt
    # fallback: single content field
    if 'content' in rec:
        return rec['content'].strip()
    # generic fallback
    return json.dumps(rec, ensure_ascii=False)


def find_lora_target_modules(model) -> List[str]:
    # Common target names in many causal models
    candidates = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'qkv_proj', 'query_key_value', 'attn.q_proj']
    names = set()
    for name, module in model.named_modules():
        for c in candidates:
            if name.endswith(c) or c in name:
                # take the final module token as target
                parts = name.split('.')
                if parts:
                    names.add(parts[-1])
    if not names:
        # fallback to linear layers
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'Linear' or module.__class__.__name__ == 'Conv1D':
                parts = name.split('.')
                names.add(parts[-1])
    return list(names)


def tokenize_and_group(tokenizer, texts: List[str], max_length: int):
    enc = tokenizer(texts, truncation=True, padding='longest', max_length=max_length)
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to training JSONL')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_jsonl(args.data_file)
    if not records:
        raise SystemExit('No records found in data file')
    texts = [build_text_from_record(r) for r in records]

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit via bitsandbytes if available
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,
        device_map='auto',
    )

    # Prepare k-bit training
    model = prepare_model_for_kbit_training(model)

    # Determine target modules for LoRA
    target_modules = find_lora_target_modules(model)
    print('Detected LoRA target modules:', target_modules)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    # Build dataset
    enc = tokenize_and_group(tokenizer, texts, args.max_seq_length)
    dataset = Dataset.from_dict(enc)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=50,
        save_steps=500,
        optim='paged_adamw_8bit',
        remove_unused_columns=False,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_dir / 'lora_adapter')
    print('LoRA adapter saved to', output_dir / 'lora_adapter')


if __name__ == '__main__':
    main()
