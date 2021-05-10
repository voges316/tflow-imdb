#!/usr/bin/env python3

import argparse
import glob
import pathlib
import re

# Lemmatize with POS Tag
import nltk
import numpy as np
import pandas as pd
# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# To eliminate stop word we need to download its vocab from nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

########################################################################################################
# Process text file to np array section
# These don't have anything to do with the rest of the script, but this info is difficult to track down,
# So I'm leaving it here as a reference on how to read in aclImdb folder contents into numpy arrays
########################################################################################################


def get_text_list_from_files(files):
    """Read folder file contents into list"""
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):
    """Read aclImdb train/test folder return pandas dataframe of contents"""
    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)

    return df


def processing_text_files_to_np_array():
    """This takes a long time, single threaded and all :( """
    train_df = get_data_from_text_files("train")
    test_df = get_data_from_text_files("test")

    train_x = []

    for r in range(len(train_df["review"])):
        if (r + 1) % 1000 == 0:
            print("No of reviews processed =", r + 1)
        train_x.append(process_text_adv(train_df["review"][r]))

    # creating train_y as target variable in the form of an array
    train_y = np.array(train_df["sentiment"])

    test_x = []

    for r in range(len(test_df["review"])):
        if (r + 1) % 1000 == 0:
            print("No of reviews processed =", r + 1)
        test_x.append(process_text_adv(test_df["review"][r]))

    test_y = np.array(test_df["sentiment"])

    return (train_x, train_y), (test_x, test_y)


########################################################################################################
# End of process text file to np array section
########################################################################################################


def cleanup_text_files_in_folder(folder_name):
    text_files = []

    for file_path in pathlib.Path(folder_name).glob('*.txt'):
        text_files.append(str(file_path))

    print(f'Found {len(text_files)} files in {folder_name}')

    # Give some kind of status
    i = 0
    for text_file in text_files:
        replace_file_contents(text_file)
        i += 1

        if i % 1000 == 0:
            print("No of files processed =", i)

    return text_files


def replace_file_contents(input_file):
    """
    This will read in the contents of the text file, process it (clean up, remove stop words)
    and overwrite the new 'processed' output to that same file

    """
    with open(input_file, 'r') as file:
        file_data = file.read()

    file_data = process_text_adv(file_data)

    with open(input_file, 'w') as file:
        file.write(file_data)


def process_text_adv(text):
    # review without HTML tags
    text = BeautifulSoup(text, features="html.parser").get_text()

    # review without punctuation and numbers
    text = re.sub("[^a-zA-Z]", ' ', text)
    # ?? Same thing, I think
    # text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    # lowercase
    text = text.lower()

    # simplest
    text = text.split()

    # simple
    # text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    # text = [lemmatizer.lemmatize(token, "v") for token in text]

    # advanced. takes longer
    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    # text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]

    # review without stopwords
    swords = set(stopwords.words("english"))  # conversion into set for fast searching
    text = [w for w in text if w not in swords]

    # joining of splitted paragraph by spaces and return
    return " ".join(text)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="""
            Script to get all files in a directory and clean them up in preparation for tensorflow processing

            Basic run puts results in an output folder
            ./imdb_parser.py --dir aclImdb/train/pos
            ./imdb_parser.py --dir aclImdb/train/neg
            ./imdb_parser.py --dir aclImdb/test/pos
            ./imdb_parser.py --dir aclImdb/test/neg

            """, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-d', '--dir', dest='default_dir', required=True,
                   help='directory folder to parse and cleanup txt files')
    args = p.parse_args()

    cleanup_text_files_in_folder(args.default_dir)

    print('Complete')
