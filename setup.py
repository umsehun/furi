from setuptools import setup, find_packages

setup(
    name="qwen_train",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.1",
        "accelerate>=0.24.0",
        "trl>=0.7.4",
        "datasets>=2.14.0",
        "kss>=3.7.0",
    ],
)
