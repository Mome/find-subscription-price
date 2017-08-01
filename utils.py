import os
import re

def find_latest_model(models_dir='./models'):
    newes = None
    time = 0.0
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        this_time = os.path.getctime(path)
        if this_time > time:
            time = this_time
            newest = path
    return path


def render_enum(seq, conj='and'):
    seq = tuple(seq)
    if len(seq) == 0:
        return ''
    if len(seq) == 1:
        return seq[0]
    head = ', '.join(seq[:-1]) 
    tail = seq[-1]
    return ' '.join([head, conj, tail])


def has_negation(sent):
    # maybe add sentiment analysis here ...
    return ('not' in sent)


def normalize_text(text):
    text = text.lower()
    
    # replace synonyms
    words = (synonyms.get(w, w) for w in nlp.tokenize(text))
    
    return ' '.join(text)


def find_numbers(line):
    pattern = r"(\d+(?:[\.,]\d*)?)"
    groups = re.findall(pattern, line)
    return groups