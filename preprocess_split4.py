"""Korean dialogue text preprocessor with sequential file processing."""
import argparse
from pathlib import Path
import json
import re
import logging
import time
from typing import List, Dict, Any, Generator
from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing."""
    min_length: int = 10  # 최소 10자
    max_length: int = 150  # 최대 150자로 늘림
    remove_urls: bool = True
    output_prefix: str = "f.dataset"

class TextPreprocessor:
    """Sequential text preprocessor for Korean dialogue data."""
    
    def __init__(self, config: PreprocessConfig):
        """Initialize preprocessor with config."""
        self.config = config
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.noise_pattern = re.compile(r'[^\w\s가-힣.,!?()\'"-]')
        self.space_pattern = re.compile(r'\s+')
        # 개선된 문장 패턴: 따옴표 내부와 일반 문장, 중간 구두점도 포함
        self.sentence_pattern = re.compile(
            r'(?:"[^"]*"(?:[.!?]|\s*$)|[^.!?"]+(?:[.!?]|$))'
        )
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing noise and normalizing."""
        if not text:
            return ""
            
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove noise and normalize spaces
        text = self.noise_pattern.sub('', text)
        text = self.space_pattern.sub(' ', text)
        
        # Remove timestamps (00:00 format)
        text = re.sub(r'\d{2}:\d{2}', '', text)
        
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better handling of quotes and punctuation."""
        paragraphs = text.split('\n')
        sentences = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 먼저 명확한 문장 끝을 찾음
            matches = self.sentence_pattern.finditer(para)
            current_sentence = []
            
            for match in matches:
                sent = match.group().strip()
                if not sent:
                    continue
                    
                # 문장이 너무 짧으면 다음 문장과 결합
                if len(sent) < self.config.min_length and current_sentence:
                    current_sentence.append(sent)
                else:
                    if current_sentence:
                        combined = ' '.join(current_sentence)
                        if not re.search(r'[.!?]$', combined):
                            combined += '.'
                        sentences.append(combined)
                        current_sentence = []
                    
                    if len(sent) >= self.config.min_length:
                        if not re.search(r'[.!?]$', sent):
                            sent += '.'
                        sentences.append(sent)
                    else:
                        current_sentence.append(sent)
            
            # 남은 문장 처리
            if current_sentence:
                combined = ' '.join(current_sentence)
                if not re.search(r'[.!?]$', combined):
                    combined += '.'
                sentences.append(combined)
        
        return sentences

    def split_long_sentence(self, sent: str) -> List[str]:
        """Split overly long sentences into smaller chunks by clause/punctuation.
        Tries commas and common Korean conjunctions; fallbacks to slicing.
        """
        max_len = self.config.max_length
        sent_ns = sent.replace(' ', '')
        if len(sent_ns) <= max_len:
            return [sent]

        # Try splitting by commas, similar punctuation or newlines
        parts = re.split(r'[,，;；·—\n]+', sent)
        # If splitting produced little, try Korean conjunctions
        if len(parts) <= 1:
            parts = re.split(r'(그런데|하지만|그리고|그리고도|또는|그러면|그럼|근데|그래서)', sent)

        # Combine into chunks not exceeding max_len
        chunks = []
        current = ''
        for p in [p.strip() for p in parts if p and p.strip()]:
            candidate = (current + ' ' + p).strip() if current else p
            if len(candidate.replace(' ', '')) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part itself too long, slice it
                if len(p.replace(' ', '')) > max_len:
                    raw = p.replace('\n', ' ')
                    # slice by characters (preserve spaces roughly)
                    idx = 0
                    clean = raw
                    while idx < len(clean):
                        piece = clean[idx: idx + max_len]
                        chunks.append(piece.strip())
                        idx += max_len
                    current = ''
                else:
                    current = p

        if current:
            chunks.append(current)

        # Ensure punctuation at end
        for i, c in enumerate(chunks):
            if not re.search(r'[.!?]$', c):
                chunks[i] = c + '.'

        return chunks
    
    def is_valid_sentence(self, text: str) -> bool:
        """Enhanced sentence validation with better Korean ratio check."""
        if not text:
            return False
            
        # Remove spaces for length check
        text_no_space = text.replace(' ', '')
        text_len = len(text_no_space)
        
        # Check length
        if text_len < self.config.min_length or text_len > self.config.max_length:
            return False
            
        # Count Korean characters
        hangul_count = len(re.findall(r'[가-힣]', text))
        
        # Ensure Korean content (최소 5개의 한글 문자)
        if hangul_count < 5:
            return False
            
        # Must end with proper punctuation
        if not re.search(r'[!?.]$', text):
            return False
            
        # 한글 비율 체크 (35% 이상, 공백 제외)
        if hangul_count / text_len < 0.35:
            return False
            
        # 연속된 구두점 체크
        if re.search(r'[!?.,]{3,}', text):
            return False
            
        return True
    
    def create_sample(self, text: str) -> Dict[str, Any]:
        """Create a simple text sample."""
        return {"text": text}
    
    def get_output_path(self, output_dir: Path, index: int) -> Path:
        """Generate output path for a sentence."""
        if index == 1:
            return output_dir / f"{self.config.output_prefix}.jsonl"
        return output_dir / f"{self.config.output_prefix}_{index}.jsonl"
    
    def save_sentence(self, sentence: str, output_path: Path) -> bool:
        """Save a single sentence to a JSONL file."""
        try:
            sample = self.create_sample(sentence)
            with open(output_path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(sample, ensure_ascii=False)
                f.write(json_str + '\n')
            return True
        except Exception as e:
            logging.error(f"Failed to save {output_path}: {e}")
            return False
    
    def process_file(
        self,
        file_path: Path,
        output_dir: Path,
        start_index: int,
        show_samples: bool = True,
        existing_texts: set = None
    ) -> int:
        """Process a single text file and return next index."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")
            return start_index
        
        # Clean and split text
        text = self.clean_text(text)
        sentences = self.split_sentences(text)
        
        # Filter and save valid sentences
        current_index = start_index
        valid_count = 0
        total_count = len(sentences)
        
        logging.info(f"\nProcessing {file_path.name} ({total_count} sentences)...")
        
        for sent in sentences:
            sent = sent.strip()
            # If too long, split into chunks
            candidates = [sent]
            sent_ns = sent.replace(' ', '')
            if len(sent_ns) > self.config.max_length:
                candidates = self.split_long_sentence(sent)

            for c in candidates:
                c = c.strip()
                if not c:
                    continue
                # Skip duplicate if already in existing_texts
                if existing_texts is not None and c in existing_texts:
                    logging.debug(f"Skipping duplicate sentence: {c[:40]}...")
                    continue

                if self.is_valid_sentence(c):
                    output_path = self.get_output_path(output_dir, current_index)
                    
                    if self.save_sentence(c, output_path):
                        # Add to existing_texts to avoid duplicates within same run
                        if existing_texts is not None:
                            existing_texts.add(c)
                        # Print detailed info for first few samples
                        if show_samples and valid_count < 3:
                            logging.info(
                                f"[{current_index}] Length: {len(c)}, "
                                f"File: {output_path.name}\n"
                                f"Text: {c}\n"
                            )
                        elif valid_count % 10 == 0:  # Progress update
                            logging.info(
                                f"[{current_index}] Generated: {output_path.name}"
                            )
                        
                        valid_count += 1
                        current_index += 1
        
        logging.info(
            f"Completed {file_path.name}: "
            f"{valid_count} valid sentences from {total_count} total\n"
            f"{'=' * 50}\n"
        )
        return current_index

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Korean dialogue preprocessor (sequential processing)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='src/data/txt',
        help='Directory containing input text files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='Minimum sentence length'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=150,
        help='Maximum sentence length'
    )
    parser.add_argument(
        '--files',
        type=str,
        default='',
        help='Comma-separated filenames or indices to process (e.g. "1,3-5" or "clo_f.txt,hagun_f.txt"). If empty, script will prompt interactively.'
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Initialize preprocessor
    config = PreprocessConfig(
        min_length=args.min_length,
        max_length=args.max_length
    )
    processor = TextPreprocessor(config)
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List available txt files in input directory
    available = sorted([p.name for p in input_dir.glob('*.txt')])
    if not available:
        logging.warning(f'No .txt files found in {input_dir}')
        return

    # Helper to parse selection strings like "1,3-5" or comma-separated names
    def parse_selection(sel: str, items: list) -> list:
        sel = sel.strip()
        if not sel:
            return []
        sel = sel.replace(' ', '')
        if sel.lower() == 'all':
            return items[:]

        chosen = []
        parts = sel.split(',')
        for part in parts:
            if '-' in part:
                try:
                    a, b = part.split('-', 1)
                    i = int(a) - 1
                    j = int(b) - 1
                    for idx in range(min(i, j), max(i, j) + 1):
                        if 0 <= idx < len(items):
                            chosen.append(items[idx])
                except Exception:
                    # treat as literal name fallback
                    if part in items:
                        chosen.append(part)
            else:
                # numeric index?
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(items):
                        chosen.append(items[idx])
                else:
                    # filename
                    if part in items:
                        chosen.append(part)
        # preserve order and remove duplicates
        result = []
        for x in chosen:
            if x not in result:
                result.append(x)
        return result

    # Determine selected files (non-interactive via --files or interactive prompt)
    selected = []
    if args.files:
        selected = parse_selection(args.files, available)
        if not selected:
            logging.error('No valid files parsed from --files argument.')
            return
    else:
        # Show list and prompt user
        print('\nAvailable .txt files:')
        for i, name in enumerate(available, 1):
            print(f"  {i}. {name}")
        print('\nChoose files to process in order. Examples:')
        print('  - all')
        print('  - 1,3,5')
        print('  - 2-4')
        print('  - clo_f.txt,hagun_f.txt')
        sel = input('\nEnter selection: ').strip()
        selected = parse_selection(sel, available)
        if not selected:
            logging.error('No files selected. Exiting.')
            return

    # Determine start index from existing output files to avoid overwriting
    existing = sorted(output_dir.glob(f"{config.output_prefix}*.jsonl"))
    if existing:
        max_idx = 0
        for p in existing:
            name = p.stem  # e.g. f.dataset_12
            parts = name.split('_')
            if len(parts) == 1:
                # base file: f.dataset
                idx = 1
            else:
                try:
                    idx = int(parts[-1])
                except Exception:
                    idx = 0
            if idx > max_idx:
                max_idx = idx
        current_index = max_idx + 1
        logging.info(f"Resuming output numbering at {current_index} (found {len(existing)} existing files)")
    else:
        current_index = 1
    # Collect existing texts to avoid duplicates
    existing_texts = set()
    for p in existing:
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict) and 'text' in obj:
                            existing_texts.add(obj['text'].strip())
                    except Exception:
                        continue
        except Exception:
            continue
    total_files = len(selected)
    logging.info(f"Processing {total_files} selected files")
    logging.info('=' * 50)

    for i, filename in enumerate(selected, 1):
        file_path = input_dir / filename
        if not file_path.exists():
            logging.error(f"File not found: {filename}")
            continue

        logging.info(f"\nFile {i}/{total_files}: {filename}")
        current_index = processor.process_file(
            file_path,
            output_dir,
            current_index,
            show_samples=(i == 1),  # 첫 번째 파일만 샘플 표시
            existing_texts=existing_texts
        )
        time.sleep(0.5)
    
    total_sentences = current_index - 1
    if total_sentences > 0:
        logging.info(f"\nProcessing completed!")
        logging.info(f"Total JSONL files created: {total_sentences}")
        logging.info(f"Files saved in: {output_dir}")
        logging.info("\nFile naming format:")
        logging.info(f"  - First file:  {config.output_prefix}.jsonl")
        logging.info(f"  - Last file:   {config.output_prefix}_{total_sentences}.jsonl")
        
        # Count files for verification
        actual_files = len(list(output_dir.glob("*.jsonl")))
        if actual_files != total_sentences:
            logging.warning(
                f"Warning: Expected {total_sentences} files "
                f"but found {actual_files}"
            )
    else:
        logging.warning("No valid sentences found!")

if __name__ == '__main__':
    main()
