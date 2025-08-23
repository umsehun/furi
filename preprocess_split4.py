"""Korean dialogue text preprocessor with sequential file processing."""
import argparse
from pathlib import Path
import json
import re
import logging
import concurrent.futures
import multiprocessing
import time
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
import concurrent.futures
import warnings
# Defer importing kss until needed to avoid kss emitting messages at module import time.
_kss = None


def _kss_split_worker(text: str, backend: str = ''):
    """Module-level worker to call kss.split_sentences in a subprocess (picklable).
    It attempts to suppress kss messages by temporarily redirecting stdout/stderr.
    """
    try:
        import kss as _kss_local  # type: ignore
        import sys, os
        null = open(os.devnull, 'w')
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = null
            sys.stderr = null
            try:
                if backend:
                    return _kss_local.split_sentences(text, backend=backend)
                return _kss_local.split_sentences(text)
            except TypeError:
                return _kss_local.split_sentences(text)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            null.close()
    except Exception:
        raise


def iter_chunks(text: str, max_len: int = 2000):
    """Yield smaller chunks from text to avoid backend blowups.
    Strategy: split by double-newline paragraphs, then by sentence punctuation
    to keep chunks under max_len characters.
    """
    if not text:
        return
    # First split on double newlines (paragraphs)
    paras = re.split(r'\n{2,}', text)
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            yield para
            continue

        # Fallback: split by sentence punctuation while preserving separators
        parts = re.split(r'([.!?。！？\n])', para)
        buf = ''
        for seg in parts:
            buf += seg
            if len(buf) >= max_len:
                yield buf
                buf = ''
        if buf:
            yield buf


@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing."""
    min_length: int = 10  # 최소 10자
    max_length: int = 150  # 최대 150자로 늘림
    remove_urls: bool = True
    output_prefix: str = "f.dataset"
    use_kss: bool = False
    kss_timeout: int = 15  # seconds to wait for kss before falling back
    kss_backend: str = ''
    kss_chunk_size: int = 2000

class TextPreprocessor:
    """Sequential text preprocessor for Korean dialogue data."""

    def __init__(self, config: PreprocessConfig):
        """Initialize preprocessor with config."""
        self.config = config

        # optional kss sentence splitter (lazy import handled elsewhere)
        self.kss = _kss if (self.config.use_kss and _kss is not None) else None

        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.noise_pattern = re.compile(r"[^\w\s가-힣.,!?()'\"-]")
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
        """Enhanced sentence splitting with better handling of quotes and punctuation.
        If kss is enabled and available, use it for higher-quality splitting.
        """
        # Use kss if available, but process the input in smaller chunks and
        # run each chunk in a short-lived subprocess to avoid pecab overflow/hangs.
        if self.kss:
            sents = []
            timeout = getattr(self.config, 'kss_timeout', 15)
            chunk_size = getattr(self.config, 'kss_chunk_size', 2000)
            backend = getattr(self.config, 'kss_backend', '')
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exe:
                for chunk in iter_chunks(text, max_len=chunk_size):
                    future = exe.submit(_kss_split_worker, chunk, backend)
                    try:
                        raw = future.result(timeout=timeout)
                        for s in raw:
                            s = s.strip()
                            if not s:
                                continue
                            if not re.search(r'[.!?]$', s):
                                s = s + '.'
                            sents.append(s)
                    except concurrent.futures.TimeoutError:
                        logging.warning(
                            f"KSS split_sentences timed out after {timeout} seconds on a chunk; falling back to regex for that chunk"
                        )
                        sents.extend(self._regex_split(chunk))
                    except Exception as e:
                        logging.warning(f'KSS processing raised exception for chunk: {e}; falling back to regex')
                        sents.extend(self._regex_split(chunk))

            return sents
        # Fallback regex path: split paragraphs into sentences
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

        # Now post-process the extracted sentences with aggressive splitting
        sents = sentences
        processed = []
        # Heuristics: if sentence length > 40% of max_length, or contains many commas/conjunctions
        comma_thresh = 1
        conj_re = re.compile(r'그런데|하지만|그리고|근데|그래서|그러면')
        max_len = self.config.max_length
        for s in sents:
            clean_ns_len = len(s.replace(' ', ''))
            comma_count = s.count(',') + s.count('，')
            if clean_ns_len > int(max_len * 0.4) or comma_count >= comma_thresh or conj_re.search(s):
                parts = self._aggressive_split(s)
                # If aggressive split returned only a single part but sentence is still
                # clause-dense (above a higher threshold), force a midpoint split.
                if len(parts) == 1 and clean_ns_len > int(max_len * 0.6):
                    raw = parts[0]
                    # find a split near the middle as a last resort
                    raw_ns = raw.replace(' ', '')
                    half = len(raw_ns) // 2
                    search_window = 40
                    start = max(0, half - search_window)
                    end = min(len(raw), half + search_window)
                    split_pos = None
                    for i in range(start, end):
                        if raw[i] in ',，;；.。!?':
                            split_pos = i + 1
                            break
                    if split_pos is None:
                        for i in range(end - 1, start - 1, -1):
                            if raw[i].isspace():
                                split_pos = i
                                break
                    if split_pos:
                        a = raw[:split_pos].strip()
                        b = raw[split_pos:].strip()
                        parts = []
                        if a:
                            parts.append(a if re.search(r'[.!?]$', a) else a + '.')
                        if b:
                            parts.append(b if re.search(r'[.!?]$', b) else b + '.')
                processed.extend(parts)
            else:
                processed.append(s)

        return processed

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
                    # Try splitting by spaces (word-aware) to avoid breaking words
                    words = p.replace('\n', ' ').split()
                    piece = ''
                    for w in words:
                        candidate = (piece + ' ' + w).strip() if piece else w
                        if len(candidate.replace(' ', '')) <= max_len:
                            piece = candidate
                        else:
                            if piece:
                                chunks.append(piece.strip())
                            # if single word too long, fallback to char slice
                            if len(w.replace(' ', '')) > max_len:
                                raw = w
                                idx = 0
                                while idx < len(raw):
                                    chunk_piece = raw[idx: idx + max_len]
                                    chunks.append(chunk_piece.strip())
                                    idx += max_len
                                piece = ''
                            else:
                                piece = w
                    if piece:
                        chunks.append(piece.strip())
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

    def _aggressive_split(self, sent: str) -> List[str]:
        """Try several lightweight, high-recall splits to break long monologues.
        This is used after KSS/regex when sentences are still too long.
        """
        # 1) Strong punctuation split: ., ?, !, ~ (and fullwidth variants)
        strong_re = re.compile(r'([\.\?!~。！？])')
        tokens = [t for t in re.split(strong_re, sent) if t and t.strip()]
        # recombine punctuation with previous fragment
        segs = []
        buf = ''
        for tok in tokens:
            if strong_re.fullmatch(tok):
                # punctuation token: append to buffer and flush
                buf = (buf + tok).strip()
                segs.append(buf)
                buf = ''
            else:
                if buf:
                    segs.append(buf)
                buf = tok
        if buf:
            segs.append(buf)

        # if we got multiple strong-punct segments, return them (shortened)
        if len(segs) > 1:
            out = []
            for s in segs:
                s = s.strip()
                if not re.search(r'[\.\?!~]$', s):
                    s = s + '.'
                out.append(s)
            return out

        # 2) Conjunction-aware split: split BEFORE conjunction and attach the conjunction
        #    to the start of the following segment (so '... 그런데 ...' -> ['...', '그런데 ...'])
        # 2a) Heuristic: split when a predicate-like ending is followed by a subject-marked noun
        # e.g. '...속수무책이네 마비아가 가져온...' -> split between '이네' and '마비아가'
        subj_change = re.search(r"(.{5,}?(?:이네|네요|습니다|다|요|군요|구나))\s+([가-힣]+(?:이|가|는|은)\b.*)", sent)
        if subj_change:
            a = subj_change.group(1).strip()
            b = subj_change.group(2).strip()
            parts = []
            if a:
                if not re.search(r'[\.\?!~]$', a):
                    a = a + '.'
                parts.append(a)
            if b:
                if not re.search(r'[\.\?!~]$', b):
                    b = b + '.'
                parts.append(b)
            if len(parts) > 1:
                return parts

        conj_words = ['근데', '그리고', '하지만', '그래서', '그런데', '그러면']
        conj_pattern = '(' + '|'.join(map(re.escape, conj_words)) + ')'
        parts = [sent]
        new_parts = []
        for p in parts:
            if len(p.replace(' ', '')) <= self.config.max_length:
                new_parts.append(p)
                continue
            toks = [t for t in re.split(conj_pattern, p) if t and t.strip()]
            if len(toks) <= 1:
                new_parts.append(p)
                continue
            i = 0
            buf = ''
            while i < len(toks):
                t = toks[i]
                if t in conj_words:
                    # attach to next
                    if i + 1 < len(toks):
                        nxt = toks[i + 1].strip()
                        if buf:
                            new_parts.append(buf.strip())
                            buf = ''
                        new_parts.append((t + ' ' + nxt).strip())
                        i += 2
                    else:
                        # conj at end
                        buf = (buf + ' ' + t).strip()
                        i += 1
                else:
                    if buf:
                        buf = (buf + ' ' + t).strip()
                    else:
                        buf = t
                    i += 1
            if buf:
                new_parts.append(buf.strip())
        parts = new_parts if new_parts else parts

        # 3) Clause-marker heuristic split (meaning-unit): split around common clause endings
        clause_markers = ['는데', '지만', '면서', '면서도', '아서', '어서', '도록', '하면', '어서도']
        out = []
        for p in parts:
            if len(p.replace(' ', '')) <= self.config.max_length:
                out.append(p)
                continue
            toks = [t for t in re.split('(' + '|'.join(map(re.escape, clause_markers)) + ')', p) if t and t.strip()]
            if len(toks) <= 1:
                out.append(p)
                continue
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i + 1] in clause_markers:
                    merged = (toks[i] + toks[i + 1]).strip()
                    out.append(merged)
                    i += 2
                else:
                    out.append(toks[i].strip())
                    i += 1

        # enforce max length and final punctuation; if still too long, midpoint-split fallback
        final = []
        for seg in out:
            seg = seg.strip()
            if not seg:
                continue
            if len(seg.replace(' ', '')) <= self.config.max_length:
                if not re.search(r'[\.\?!~]$', seg):
                    seg = seg + '.'
                final.append(seg)
                continue

            # midpoint split near punctuation or whitespace
            raw = seg
            raw_ns = raw.replace(' ', '')
            half = len(raw_ns) // 2
            search_window = 40
            start = max(0, half - search_window)
            end = min(len(raw), half + search_window)
            split_pos = None
            for i in range(start, end):
                if raw[i] in ',，;；.。.。!?~':
                    split_pos = i + 1
                    break
            if split_pos is None:
                for i in range(end - 1, start - 1, -1):
                    if raw[i].isspace():
                        split_pos = i
                        break
            if split_pos is None:
                idx = 0
                while idx < len(raw):
                    piece = raw[idx: idx + self.config.max_length].strip()
                    if piece:
                        if not re.search(r'[\.\?!~]$', piece):
                            piece = piece + '.'
                        final.append(piece)
                    idx += self.config.max_length
            else:
                a = raw[:split_pos].strip()
                b = raw[split_pos:].strip()
                if a:
                    if not re.search(r'[\.\?!~]$', a):
                        a = a + '.'
                    final.append(a)
                if b:
                    if not re.search(r'[\.\?!~]$', b):
                        b = b + '.'
                    final.append(b)

        return final

    
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

    def _regex_split(self, text: str) -> List[str]:
        """Fallback regex-based splitter used when kss fails or times out."""
        paragraphs = text.split('\n')
        sentences: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            matches = self.sentence_pattern.finditer(para)
            current_sentence = []
            for match in matches:
                sent = match.group().strip()
                if not sent:
                    continue
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
            if current_sentence:
                combined = ' '.join(current_sentence)
                if not re.search(r'[.!?]$', combined):
                    combined += '.'
                sentences.append(combined)
        return sentences

    def build_turns(self, sentences: List[str], max_sent_per_turn: int = 2) -> List[str]:
        """Group adjacent sentences into conversational 'turns'.
        This reduces mismatch when a single content block lacks explicit speaker labels.
        """
        turns: List[str] = []
        buf: List[str] = []
        for s in sentences:
            buf.append(s)
            # if buffer reached capacity or sentence ends with strong punctuation, flush
            if len(buf) >= max_sent_per_turn:
                turns.append(' '.join(buf).strip())
                buf = []
        if buf:
            turns.append(' '.join(buf).strip())
        # Filter out turns that are too short
        good_turns = [t for t in turns if len(re.sub(r'\s+', '', t)) >= 5]
        return good_turns
    
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
        existing_texts: set = None,
        content_path: Path = None,
        pair_path: Path = None
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
        valid_sentences = []

        logging.info(f"\nProcessing {file_path.name} ({total_count} sentences)...")

        for sent in sentences:
            sent = sent.strip()
            # If too long, split into chunks
            candidates = [sent]
            sent_ns = sent.replace(' ', '')
            if len(sent_ns) > self.config.max_length:
                # First try clause-aware splitting
                candidates = self.split_long_sentence(sent)
                # If still too long, try aggressive splitting
                new_cands = []
                for cc in candidates:
                    if len(cc.replace(' ', '')) > self.config.max_length:
                        new_cands.extend(self._aggressive_split(cc))
                    else:
                        new_cands.append(cc)
                candidates = new_cands

            for c in candidates:
                c = c.strip()
                if not c:
                    continue
                # Skip duplicate if already in existing_texts
                if existing_texts is not None and c in existing_texts:
                    logging.debug(f"Skipping duplicate sentence: {c[:40]}...")
                    continue

                if self.is_valid_sentence(c):
                    # collect valid sentences for aggregated outputs
                    valid_sentences.append(c)

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
        # Append valid sentences to aggregated content file
        if content_path and valid_sentences:
            try:
                with open(content_path, 'a', encoding='utf-8') as cf:
                    for s in valid_sentences:
                        cf.write(json.dumps({"role":"content","content":s}, ensure_ascii=False) + '\n')
            except Exception as e:
                logging.error(f"Failed to append to content file {content_path}: {e}")

        # Create adjacent user->assistant pairs and append
        if pair_path and len(valid_sentences) >= 2:
            try:
                # Group sentences into conversational turns to reduce mismatch
                turns = self.build_turns(valid_sentences, max_sent_per_turn=2)

                # If only one turn exists, try to split it into two parts to form a pair
                if len(turns) == 1:
                    lone = turns[0]
                    lone_ns = len(lone.replace(' ', ''))
                    # Prefer aggressive split
                    parts = self._aggressive_split(lone)
                    if len(parts) == 1 and lone_ns > int(self.config.max_length * 0.5):
                        # force a midpoint split
                        raw = parts[0]
                        raw_ns = raw.replace(' ', '')
                        half = len(raw_ns) // 2
                        search_window = 40
                        start = max(0, half - search_window)
                        end = min(len(raw), half + search_window)
                        split_pos = None
                        for i in range(start, end):
                            if raw[i] in ',，;；.。!?':
                                split_pos = i + 1
                                break
                        if split_pos is None:
                            for i in range(end - 1, start - 1, -1):
                                if raw[i].isspace():
                                    split_pos = i
                                    break
                        if split_pos:
                            a = raw[:split_pos].strip()
                            b = raw[split_pos:].strip()
                            parts = []
                            if a:
                                parts.append(a if re.search(r'[.!?]$', a) else a + '.')
                            if b:
                                parts.append(b if re.search(r'[.!?]$', b) else b + '.')

                    # If we have at least two parts, build pair from first two
                    if len(parts) >= 2:
                        with open(pair_path, 'a', encoding='utf-8') as pf:
                            pair_obj = {"user": parts[0], "assistant": parts[1]}
                            pf.write(json.dumps(pair_obj, ensure_ascii=False) + '\n')
                    else:
                        # fallback: no pair created
                        logging.debug('Single turn present but unable to split into pair; skipping pair creation')
                else:
                    with open(pair_path, 'a', encoding='utf-8') as pf:
                        for a, b in zip(turns, turns[1:]):
                            pair_obj = {"user": a, "assistant": b}
                            pf.write(json.dumps(pair_obj, ensure_ascii=False) + '\n')
            except Exception as e:
                logging.error(f"Failed to append to pair file {pair_path}: {e}")

        return current_index

def main():
    """Main execution function."""
    global _kss
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
    parser.add_argument(
        '--use_kss',
        action='store_true',
        help='Use KSS for sentence splitting if available'
    )
    parser.add_argument(
        '--force-regex',
        dest='force_regex',
        action='store_true',
        help='Force regex-based splitting and disable KSS even if --use_kss is passed'
    )
    parser.add_argument(
        '--content_file',
        type=str,
        default='content_only.jsonl',
        help='Aggregated content-only output JSONL filename'
    )
    parser.add_argument(
        '--pair_file',
        type=str,
        default='pairs.jsonl',
        help='Aggregated user-assistant pair output JSONL filename'
    )
    parser.add_argument(
        '--start-index',
        dest='start_index',
        type=int,
        default=None,
        help='Force starting index (1-based). If provided, numbering will start from this value and not resume from existing files.'
    )
    parser.add_argument(
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='When set, do not skip sentences found in existing outputs and allow writing from the start index (may overwrite files).'
    )
    parser.add_argument(
        '--kss-backend',
        dest='kss_backend',
        type=str,
        default='',
        help='Preferred kss backend (e.g. "mecab"). If empty, kss will auto-detect.'
    )
    parser.add_argument(
        '--kss-chunk-size',
        dest='kss_chunk_size',
        type=int,
        default=2000,
        help='Max characters per chunk fed to KSS to avoid backend blowups.'
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Initialize preprocessor
    # If user requests force-regex, disable kss regardless of --use_kss
    if getattr(args, 'force_regex', False):
        logging.info('Force-regex enabled: KSS will be ignored and regex splitting will be used')

    config = PreprocessConfig(
        min_length=args.min_length,
        max_length=args.max_length,
        use_kss=(args.use_kss and not getattr(args, 'force_regex', False)),
        kss_backend=args.kss_backend or '',
        kss_chunk_size=getattr(args, 'kss_chunk_size', 2000)
    )
    processor = TextPreprocessor(config)
    # Lazy-load kss module only when use_kss is True and not force-regex
    if config.use_kss and _kss is None:
        try:
            import kss as _kss_mod  # type: ignore
            _kss = _kss_mod
            processor.kss = _kss
            logging.info('kss module loaded for sentence splitting')
        except Exception as e:
            logging.warning(f'Failed to import kss: {e}; falling back to regex')
    
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

    # Determine start index from existing output files to avoid overwriting,
    # unless the user forces a start index with --start-index.
    existing = sorted(output_dir.glob(f"{config.output_prefix}*.jsonl"))
    if args.start_index is not None and args.start_index > 0:
        current_index = args.start_index
        logging.info(f"Forced start index set to {current_index} by --start-index")
    else:
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
    # Collect existing texts to avoid duplicates (unless overwrite requested)
    existing_texts = set() if not getattr(args, 'overwrite', False) else None
    if existing_texts is not None:
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
    else:
        logging.info('Overwrite enabled: existing sentence deduplication disabled; files may be overwritten')
    total_files = len(selected)
    logging.info(f"Processing {total_files} selected files")
    logging.info('=' * 50)

    # Prepare aggregated output paths
    content_path = output_dir / args.content_file
    pair_path = output_dir / args.pair_file

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
            existing_texts=existing_texts,
            content_path=content_path,
            pair_path=pair_path
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
