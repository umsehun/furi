#!/bin/bash

# 가상환경 이름 설정
VENV_NAME="qwen_env"
PYTHON_VERSION="3.11.9"

# pyenv로 Python 버전 설정
if command -v pyenv 1>/dev/null 2>&1; then
    echo "Setting Python version to ${PYTHON_VERSION}..."
    pyenv install ${PYTHON_VERSION} -s
    pyenv local ${PYTHON_VERSION}
else
    echo "pyenv not found. Please install pyenv first."
    exit 1
fi

# Poetry 설치 확인
if ! command -v poetry 1>/dev/null 2>&1; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Poetry 가상환경 설정
echo "Configuring Poetry..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.create true

# 의존성 설치
echo "Installing dependencies..."
poetry install

# CUDA 버전 확인
if command -v nvidia-smi 1>/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo "Detected CUDA version: ${CUDA_VERSION}"
    
    # PyTorch CUDA 설치
    poetry run pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "NVIDIA GPU not detected. Installing CPU-only PyTorch version."
    poetry run pip3 install torch torchvision torchaudio
fi

echo "Environment setup complete!"
echo "To activate the environment, run: poetry shell"
