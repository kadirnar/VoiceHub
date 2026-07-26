import re
import uroman as ur
import MeCab
import string
import unicodedata

def check_language(text: str) -> bool:
    return any('\u3040' <= char <= '\u309f' or
                '\u30a0' <= char <= '\u30ff' or
                '\u4e00' <= char <= '\u9fff' or
                '\u4e00' <= char <= '\u9fff' and 
                any('\u3040' <= c <= '\u309f' for c in text)
                for char in text)

def get_punctuations():
    ascii_punctuation = set(string.punctuation)
    punctuation_categories = {'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps'}
    unicode_punct = set()
    
    # Check characters in common Unicode ranges
    for i in range(0x10000):  # First 65536 Unicode code points
        char = chr(i)
        category = unicodedata.category(char)
        if category in punctuation_categories:
            unicode_punct.add(char)
    
    all_punctuation = ascii_punctuation.union(unicode_punct)
    return sorted(all_punctuation)

def normalize_token_spacing(text: str):
    text = text.replace("'", "").replace("`", "")
    WHITESPACE_PATTERN = re.compile(r'(?<!\s)\s(?!\s)')
    text =  MeCab.Tagger("-Owakati").parse(text).split()
    text = [i for i in text if i not in ['', ' ']]
    result = text[0]  
    punctuations = get_punctuations()
    for token in text[1:]:
        if token not in punctuations:
            result += '  '
        else:
            result += ' '
        result += token
    return WHITESPACE_PATTERN.sub('', result)

def text_normalizations(text: str) -> str:
    # Normalize whitespace characters (newlines, tabs, etc.) to single spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("…", "...")  # Replace ellipsis character with three dots
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Normalize common Unicode characters to ASCII equivalents
    text = re.sub(r'[“”]', '"', text)   # Curly quotes to straight quotes
    text = re.sub(r'[‘’]', "'", text)   # Curly single quotes
    text = re.sub(r'[–—]', '-', text)   # Various dashes to hyphen
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    return text

def get_words(text: str):
    text = text_normalizations(text)
    text = normalize_token_spacing(text)
    if check_language(text):
        mecab = MeCab.Tagger("-Owakati")
        return mecab.parse(text).strip().split()
    return text.split()
