from bs4 import BeautifulSoup
from nltk.tokenize import regexp_span_tokenize
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
import re
from pathlib import Path

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def regexp_split(token):
    res = regexp_span_tokenize(token, re.compile("[.,-]"))
    text_len = len(token)
    
    res = list(res)
    res[0] = res[0] if res[0][0] == 0 else (0, res[0][1])
    
    tokens = []
    for x in res:
        if x[0] >= text_len:
            break
        
        tokens.append(token[x[0]: x[1]])
        
        if x[1] >= text_len:
            continue
        
        tokens.append(token[x[1]])
        
    return tokens
      
def tweet_tokenize(text):
    ttweet = TweetTokenizer()
    res = ttweet.tokenize(text)
    
    tokens = []
    for token in res:
        tokens.extend(regexp_split(token))
            
    return tokens
    

#Tokenize the text.
def tokenize_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    tokens = tweet_tokenize(text.lower())
    text = " ".join(tokens)
    return text

def read_folder(folder_name, corpus_file):
    texts_folder = Path(folder_name).rglob('*.txt')
    files = [x for x in texts_folder]
    
    for filename in tqdm(files):
        text = ''
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines()]
        
        for line in lines:
            line = tokenize_text(line)
            if len(text) > 0:
                text += ' '
            text += line
            
        corpus_file.write(text + '\n')
        corpus_file.flush()

def extract_imdb_text(folder_name: str, corpus_name: str):
    print('extract_imdb_text - start')
    with open(corpus_name, 'w', encoding='utf-8') as f:
        print("\tread test-neg")
        read_folder(folder_name + 'test/neg', f)
        print("\tread test-pos")
        read_folder(folder_name + 'test/pos', f)
        print("\tread train-neg")
        read_folder(folder_name + 'train/neg', f)
        print("\tread train-pos")
        read_folder(folder_name + 'train/pos', f)
        print("\tread train-unsup")
        read_folder(folder_name + 'train/unsup', f)
    print('extract_imdb_text - end')

def extract_elec_text(in_folder: str, out_file: str):
    print("Extract Elec texts - Start")
    texts_folder = Path(in_folder).glob('*.txt')
    files = [str(x) for x in texts_folder]
    with open(out_file, 'w', encoding='utf-8') as corpus:
        for filename in files:
            print("\tWorking on file: " + filename)
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in tqdm(lines):
                    text = tokenize_text(line)
                    corpus.write(text + "\n")
                    corpus.flush()
    print("Extract Elec texts - End")
