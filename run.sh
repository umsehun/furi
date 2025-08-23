#!/bin/bash

# 환경 활성화
echo "Activating environment..."
source .venv/bin/activate

# 필요한 패키지 설치
echo "Installing required packages..."
pip install kss

# 데이터 전처리
echo "Preprocessing data..."
PYTHONPATH=. python src/scripts/preprocess.py \
    --input_dir data/txt \
    --output_file data/dataset.jsonl \
    --min_length 4

echo "Preprocessing completed! Dataset saved to data/dataset.jsonl"

echo "Pipeline completed!"
