#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

import tensorflow as tf

# tf.keras.datasets.imdb.load_data()
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data()


def load_imdb_data(verbose=False):
    """Load the imdb reviews dataset
  This is a dataset for binary sentiment classification.
  We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing.
  There is additional unlabeled data for use as well.

  Returns:
      Tuple of tf BatchDatasets: `(raw_train_ds, raw_val_ds, raw_test_ds)`.

    """
    dataset_dir = 'aclImdb'
    if verbose:
        print(os.listdir(dataset_dir))
        # ['README', 'train', 'imdbEr.txt', 'test', 'imdb.vocab']

    train_dir = os.path.join(dataset_dir, 'train')
    if verbose:
        print(os.listdir(train_dir))
        # ['neg', 'pos', 'unsup', 'unsupBow.feat', 'labeledBow.feat', 'urls_neg.txt', 'urls_pos.txt', 'urls_unsup.txt']

    # aclImdb/train/pos and aclImdb/train/neg directories contain many text files, look at 1
    if verbose:
        sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        with open(sample_file) as f:
            print(f.read())
    '''
    Rachel Griffiths writes and directs this award winning short film.....
    '''

    # delete unsup dir, since that messes up the import
    remove_path = Path(os.path.join(train_dir, 'unsup'))
    if remove_path.exists() and remove_path.is_dir():
        shutil.rmtree(remove_path)

    '''
    dataset has already been divided into train and test, but it lacks a validation set. 
    Let's create a validation set using an 80:20 split of the training data by using the validation_split argument below
    '''
    batch_size = 1024
    seed = 123

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)
    """
    Found 25000 files belonging to 2 classes.
    Using 20000 files for training.
    """

    '''
    Next, create a validation and test dataset. 
    You will use the remaining 5,000 reviews from the training set for validation.
    '''
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)
    """
    Found 25000 files belonging to 2 classes.
    Using 5000 files for validation.
    """

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)
    """
    Found 25000 files belonging to 2 classes.
    """

    '''
    If you're new to tf.data, you can also iterate over the dataset and print out a few examples
    '''
    if verbose:
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(3):
                print("Review", text_batch.numpy()[i])
                print("Label", label_batch.numpy()[i])

    '''
    Labels are 0 or 1. To see which of these correspond to positive and negative movie reviews, 
    you can check the class_names property on the dataset
    '''
    if verbose:
        print("Label 0 corresponds to", raw_train_ds.class_names[0])
        print("Label 1 corresponds to", raw_train_ds.class_names[1])
    """
    Label 0 corresponds to neg
    Label 1 corresponds to pos
    """

    return raw_train_ds, raw_val_ds, raw_test_ds
