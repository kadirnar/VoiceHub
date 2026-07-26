import re
from typing import List
import unicodedata
import MeCab

def check_language(text: str) -> bool:
    return any('\u3040' <= char <= '\u309f' or
                '\u30a0' <= char <= '\u30ff' or
                '\u4e00' <= char <= '\u9fff' or
                '\u4e00' <= char <= '\u9fff' and 
                any('\u3040' <= c <= '\u309f' for c in text)
                for char in text)

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text based on detected language.
    """
    if not text.strip():
        return []
    
    if check_language(text):
        mecab = MeCab.Tagger("-Owakati")
        return mecab.parse(text).strip().split()
    else:
        return text.split()

def count_words(text: str) -> int:
    """Count words in text using appropriate tokenization."""
    return len(tokenize_text(text))

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using appropriate end markers."""
    # Common sentence end markers for various languages
    end_markers = r'[.!?。！？︕︖]+'
    
    # Split on sentence end markers while preserving them
    parts = re.split(f'({end_markers}\\s*)', text)
    
    sentences = []
    current_sentence = ""
    
    for i in range(0, len(parts), 2):
        current_sentence = parts[i]
        
        # Add the sentence end marker if it exists
        if i + 1 < len(parts):
            current_sentence += parts[i + 1]
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
    
    return sentences

def chunk_text(text: str, min_words: int = 10, max_words: int = 30) -> List[str]:
    """
    Chunk multilingual text into segments with specified word count bounds.
    
    Args:
        text: Input text that may contain multiple languages
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk
    
    Returns:
        List of text chunks
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Remove excessive whitespace while preserving sentence boundaries
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = tokenize_text(sentence)
        sentence_word_count = len(sentence_tokens)
        
        # Handle long sentences
        if sentence_word_count > max_words:
            # Add current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_word_count = 0
            
            # Split long sentence into parts
            current_part = []
            word_count = 0
            
            for token in sentence_tokens:
                current_part.append(token)
                word_count += 1
                
                if word_count >= max_words:
                    if check_language(sentence):
                        chunks.append("".join(current_part))
                    else:
                        chunks.append(" ".join(current_part))
                    current_part = []
                    word_count = 0
            
            if current_part:
                if check_language(sentence):
                    chunks.append("".join(current_part))
                else:
                    chunks.append(" ".join(current_part))
            continue
        
        # Try to add sentence to current chunk
        if current_word_count + sentence_word_count <= max_words:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_word_count += sentence_word_count
        else:
            # Check if current chunk meets minimum requirement
            if current_word_count >= min_words:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_word_count = sentence_word_count
            else:
                # Split the sentence to fit remaining space
                space_left = max_words - current_word_count
                current_part = sentence_tokens[:space_left]
                remaining_part = sentence_tokens[space_left:]
                
                if current_chunk:
                    if check_language(current_part):
                        chunks.append(current_chunk + "".join(current_part))
                    else:
                        chunks.append(current_chunk + " " + " ".join(current_part))
                else:
                    if check_language(current_part):
                        chunks.append("".join(current_part))
                    else:
                        chunks.append(" ".join(current_part))
                
                if check_language(remaining_part):
                    current_chunk = "".join(remaining_part)
                else:
                    current_chunk = " ".join(remaining_part)
                current_word_count = len(remaining_part)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
