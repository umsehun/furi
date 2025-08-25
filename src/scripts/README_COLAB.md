Colab QLoRA training helper

Overview
- This folder contains `colab_qloRA_train.py`, a Colab-ready script to run QLoRA training for causal LLMs (e.g., Llama 3.2 3B).
- Expects a JSONL training file (see examples below).

Colab setup (cells)
1) Install dependencies
```bash
!pip install -q transformers accelerate bitsandbytes peft datasets safetensors
```

2) Upload your training data to Colab (or mount Google Drive)
- Place the JSONL at `/content/qwen/train.jsonl` or similar.

3) Run training
```bash
!python /content/furi/furi/colab_qloRA_train.py \
  --data_file /content/qwen/train.jsonl \
  --output_dir /content/qwen_out \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 2000
```

Data format
- JSONL where each line is a JSON object. Example records supported:
  1) instruction style:
     {"instruction":"Formalize the tone","input":"야, 뭐해?","output":"안녕하세요, 무엇을 도와드릴까요?"}
  2) simple content:
     {"content":"간단한 문장 예시"}

Tips
- Start with small max_steps (e.g., 2000) and inspect outputs.
- Use `gradient_accumulation_steps` to increase effective batch without extra VRAM.
- Verify `target_modules` printed by the script and adjust if needed.

Security
- Do NOT store HF tokens in notebooks; use Colab secrets or environment variables.
