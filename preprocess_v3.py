"""Enhanced Korean dialogue preprocessor with strict formatting for LLM fine-tuning."""
import argparse
from pathlib import Path
import json
import re
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings
import numpy as np

@dataclass
class DialogueConfig:
    """Configuration for dialogue processing."""
    min_chars: int = 10  # 최소 글자수
    max_chars: int = 50  # 최대 글자수 (한글 기준)
    min_hangul_ratio: float = 0.5  # 최소 한글 비율
    samples_per_prompt: int = 10  # 프롬프트당 생성할 샘플 수
    system_message: str = "너는 한 문장으로만 대답하는 한국어 대사 생성 모델이다. 반드시 한 문장, 50자 이하, 마침표/물음표/느낌표로 끝내야 한다."
    user_message: str = "한 문장 대사로 자연스럽게 말해줘."
    output_format: str = "messages"  # 'messages' 또는 'conversations'

class KssWrapper:
    """Enhanced KSS wrapper with improved error handling."""
    
    def __init__(self):
        try:
            import kss
            self.kss = kss
            
            # Check for MeCab
            import importlib
            if importlib.util.find_spec('mecab') is None:
                logging.warning(
                    "MeCab is not installed. Using PeCab backend. "
                    "Install MeCab for better performance: brew install mecab mecab-ko mecab-ko-dic"
                )
        except ImportError as e:
            raise ImportError(f"Failed to import kss: {e}")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text with improved error handling."""
        try:
            sentences = self.kss.split_sentences(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logging.error(f"KSS splitting failed: {e}")
            # 기본 구분자로 분할 (마침표, 물음표, 느낌표)
            basic_splits = re.split(r'[.!?]+', text)
            return [s.strip() for s in basic_splits if s.strip()]

class DialogueProcessor:
    """Enhanced dialogue processor for LLM training."""
    
    def __init__(self, config: DialogueConfig):
        self.config = config
        self.kss = KssWrapper()
        
        # 정규식 패턴 컴파일
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.noise_pattern = re.compile(r'[^\w\s가-힣.,!?()\'"-]')
        self.space_pattern = re.compile(r'\s+')
        self.timestamp_pattern = re.compile(r'\d{2}:\d{2}')
        self.ending_pattern = re.compile(r'[.!?]$')
    
    def clean_text(self, text: str) -> str:
        """Clean text with enhanced filtering."""
        if not text:
            return ""
        
        # Remove URLs, timestamps, and noise
        text = self.url_pattern.sub('', text)
        text = self.timestamp_pattern.sub('', text)
        text = self.noise_pattern.sub('', text)
        
        # Normalize spaces and add ending if missing
        text = self.space_pattern.sub(' ', text).strip()
        if text and not self.ending_pattern.search(text):
            text += "."
            
        return text
    
    def is_valid_sentence(self, text: str) -> bool:
        """Enhanced sentence validation."""
        if not text:
            return False
            
        # Check length (한글 기준)
        hangul_len = len(re.findall('[가-힣]', text))
        if hangul_len < self.config.min_chars or hangul_len > self.config.max_chars:
            return False
            
        # Check Korean ratio
        total_len = len(text)
        hangul_ratio = hangul_len / total_len
        if hangul_ratio < self.config.min_hangul_ratio:
            return False
            
        # Check for proper ending
        if not self.ending_pattern.search(text):
            return False
            
        # 문장 복잡도 검사 (쉼표가 2개 이상이면 제외)
        if text.count(',') > 1:
            return False
            
        return True

    def create_dialogue_sample(self, text: str) -> Optional[Dict[str, Any]]:
        """Create a dialogue sample in the specified format."""
        if not self.is_valid_sentence(text):
            return None
            
        if self.config.output_format == "messages":
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": self.config.system_message
                    },
                    {
                        "role": "user",
                        "content": self.config.user_message
                    },
                    {
                        "role": "assistant",
                        "content": text
                    }
                ]
            }
        else:
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": self.config.system_message
                    },
                    {
                        "from": "user",
                        "value": self.config.user_message
                    },
                    {
                        "from": "assistant",
                        "value": text
                    }
                ]
            }

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file with improved sample distribution."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")
            return []
            
        # Clean and split text
        text = self.clean_text(text)
        sentences = self.kss.split_sentences(text)
        
        # Filter and create samples
        valid_samples = []
        for sent in sentences:
            sample = self.create_dialogue_sample(sent)
            if sample:
                valid_samples.append(sample)
        
        # 각 입력에 대해 여러 샘플 생성
        final_samples = []
        for _ in range(self.config.samples_per_prompt):
            if valid_samples:
                # 랜덤 선택 후 제거 (중복 방지)
                sample = valid_samples.pop(
                    np.random.randint(len(valid_samples))
                ) if valid_samples else None
                if sample:
                    final_samples.append(sample)
        
        logging.info(
            f"Processed {file_path}: {len(final_samples)} samples "
            f"from {len(sentences)} sentences"
        )
        return final_samples

def main():
    """Main execution with enhanced logging."""
    parser = argparse.ArgumentParser(
        description='Enhanced Korean dialogue preprocessor for LLM fine-tuning'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input text files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--format',
        choices=['messages', 'conversations'],
        default='messages',
        help='Output format (messages or conversations)'
    )
    parser.add_argument(
        '--samples_per_prompt',
        type=int,
        default=10,
        help='Number of samples to generate per prompt'
    )
    parser.add_argument(
        '--max_chars',
        type=int,
        default=50,
        help='Maximum characters per sentence'
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize processor
    config = DialogueConfig(
        max_chars=args.max_chars,
        samples_per_prompt=args.samples_per_prompt,
        output_format=args.format
    )
    processor = DialogueProcessor(config)
    
    # Process files
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_file)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process all text files
    all_samples = []
    for file_path in input_dir.glob('*.txt'):
        samples = processor.process_file(file_path)
        all_samples.extend(samples)
    
    if not all_samples:
        logging.warning("No valid samples were generated!")
        return
    
    # Shuffle and save
    np.random.shuffle(all_samples)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logging.info(f"Successfully saved {len(all_samples)} samples to {output_path}")
    
    # Print sample for verification
    print("\nSample output:")
    print(json.dumps(all_samples[0], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
