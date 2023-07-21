
import os
import urllib

CONLL_URL_ROOT = "https://github.com/nluninja/nlp_datasets/tree/main/GUM/data"

def open_read_from_url(url):
    """
    Take in input an url to a .txt file and return the list of its raws
    """
    print(f"Read file from {url}")
    file = urllib.request.urlopen(url)
    lines = []
    for line in file:
        lines.append(line.decode("utf-8"))

    return lines

def read_raw_conll(url_root, dir_path, filename):
    """Read a file which contains a conll03 dataset"""
    lines = []
    path = os.path.join(dir_path, filename)
    full_url = url_root + filename
    if os.path.isfile(path):
        # read from file
        print(f'Reading file {path}')
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        lines = open_read_from_url(full_url)
    return lines[1:]

def is_real_sentence(only_token, sentence):
    """Chek if a sentence is a real sentence or a document separator"""
    first_word = ""
    if only_token:
        first_word = sentence[0]
    else:
        first_word = sentence[0][0]

    if '---------------------' in first_word or first_word == '-DOCSTART-':
        return False
    else:
        return True
    
    
def load_conll_data(filename, url_root=CONLL_URL_ROOT, dir_path='', 
                    only_tokens=False):
    """
    Take an url to the raw .txt files that you can find the repo linked above,
    load data and save it into a list of tuples data structure.
    
    Those files structure data with a word in each line with word, POS, 
    syntactic tag and entity tag separated by a whitespace. Sentences are 
    separated by an empty line.
    """
    lines = read_raw_conll(url_root, dir_path, filename)
    X = []
    Y = []
    sentence = []
    labels = []
    output_labels=set()
    for line in lines:
        if line == "\n":
            if(len(sentence) != len(labels)):
                print(f"Error: we have {len(sentence)} words but {len(labels)} labels")
            if sentence and is_real_sentence(only_tokens, sentence):
                X.append(sentence)
                Y.append(labels)
            sentence = []
            labels = []
        else:
            features = line.split()
            tag = features.pop()
            labels.append(tag)
            output_labels.add(tag)
            if only_tokens:
                sentence.append(features.pop(0))
            else:
                sentence.append(tuple(features))
    
    print(f"Read {len(X)} sentences")
    if(len(X) != len(Y)):
        print("ERROR in reading data.")
    return X, Y, output_labels