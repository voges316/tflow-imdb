#!/usr/bin/env python3

import os
import re
import shutil
import string
import sys
from pathlib import Path

# https://www.tensorflow.org/tutorials/keras/text_classification
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

######################
# Parameters         #
######################
BATCH_SIZE      = 64
MAX_FEATURES    = 8500
SEQ_LENGTH      = 1000
EMBEDDING_DIM   = 8
NUM_EPOCHS      = 300

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#
# dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')
# dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

dataset_dir = 'aclImdb'
# print(os.listdir(dataset_dir))
# ['README', 'train', 'imdbEr.txt', 'test', 'imdb.vocab']

train_dir = os.path.join(dataset_dir, 'train')
# print(os.listdir(train_dir))
# ['neg', 'pos', 'unsup', 'unsupBow.feat', 'labeledBow.feat', 'urls_neg.txt', 'urls_pos.txt', 'urls_unsup.txt']

# aclImdb/train/pos and aclImdb/train/neg directories contain many text files, look at 1
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#     print(f.read())
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
batch_size = BATCH_SIZE
seed = 123

# generate tf.data.Dataset from text files in directory
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
If you're new to tf.data, you can also iterate over the dataset and print out a few examples
'''
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

'''
Labels are 0 or 1. To see which of these correspond to positive and negative movie reviews, 
you can check the class_names property on the dataset
'''
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
"""
Label 0 corresponds to neg
Label 1 corresponds to pos
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


"""
Pre-processing

Standardization - preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset.
Tokenization - splitting strings into tokens (ex, splitting sentence into individual words, by splitting on whitespace.
Vectorization - converting tokens into numbers so they can be fed into a neural network.
"""


# Next, prepare datset for training
def custom_standardization(input_data):
    """
    The reviews contain various HTML tags like <br />. These tags will not be removed by the default standardizer
    in the TextVectorization layer (which converts text to lowercase and strips punctuation by default, but doesn't
    strip HTML). Write a custom standardization function to remove the HTML
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')      # remove html tag
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), # strip punctuation
                                    '')


max_features = MAX_FEATURES
sequence_length = SEQ_LENGTH

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

''' 
Call adapt to fit the state of the preprocessing layer to the dataset. 
This will cause the model to build an index of strings to integers.
'''
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# create a function to see the result of using this layer to preprocess some data.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

"""
As you can see above, each token has been replaced by an integer. You can lookup the token (string) that each 
integer corresponds to by calling .get_vocabulary() on the layer.


Review tf.Tensor(b'Silent Night, Deadly Night 5 is .... 4 out of 5.', shape=(), dtype=string)
Label neg
Vectorized review (<tf.Tensor: shape=(1, 250), dtype=int64, numpy=
array([[1287,  313, 2380,  313,  661,    7,    2,   52,  229,    5,    2,
         ...
        4028,  948,    6,   67,   48,  158,   93,    1]])>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)

You can lookup the token (string) that each integer corresponds to by calling .get_vocabulary() on the layer.
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

1287 --->  silent
 313 --->  night
Vocabulary size: 10000
"""

# As a final preprocessing step, you will apply the TextVectorization layer you created earlier to the
# train, validation, and test dataset.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

"""
Configure dataset for performance.

Two important methods you should use when loading data to make sure that I/O does not become blocking.
.cache() keeps data in memory after it's loaded off disk.
.prefetch() overlaps data preprocessing and model execution while training. 
"""
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
embedding_dim = EMBEDDING_DIM

"""
sentiment classification model. In this case it is a "Continuous bag of words" style model
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/word_embeddings.ipynb
"""
# model = tf.keras.Sequential([
#     layers.Embedding(
#         input_dim=max_features + 1,
#         output_dim=embedding_dim),
#     layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(16, activation='relu'),
#     layers.Dense(1)
# ])

# also works without the padding input, I guess...
# model = tf.keras.Sequential([
#     layers.Embedding(
#         input_dim=max_features,
#         output_dim=embedding_dim,
#         mask_zero=True),
#     layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(16, activation='relu'),
#     layers.Dense(1)
# ])

model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim, name='embedding'),
    layers.Dropout(0.6),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(1)
])

print(model.summary())

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 8)           68000     
_________________________________________________________________
dropout (Dropout)            (None, None, 8)           0         
_________________________________________________________________
global_average_pooling1d (Gl (None, 8)                 0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8)                 0         
_________________________________________________________________
dense (Dense)                (None, 128)               1152      
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 69,281
Trainable params: 69,281
Non-trainable params: 0
_________________________________________________________________
"""

# callback to log results for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

# early stopping callback. stop training when validation loss stops decreasing
loss_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)


# Set loss function and optimizer
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.0007),
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model
epochs = NUM_EPOCHS
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback, loss_callback])

# for logging
l, a = model.evaluate(train_ds)
print('Loss & accuracy on train dataset')
print("Loss: ", l)
print("Accuracy: ", a)

# Evaluate the model. Two values will be returned.
# Loss (a number which represents our error, lower values are better), and accuracy.
loss, accuracy = model.evaluate(test_ds)

print('Loss & accuracy on test dataset')
print("Loss: ", loss)
print("Accuracy: ", accuracy)

print('Exiting...')
sys.exit()

# Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()

'''
dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])

There are four entries: one for each monitored metric during training and validation. 
You can use these to plot the training and validation loss for comparison, as well as the training and validation accuracy:
'''

# Training and validation loss
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

sleep(0.5)
plt.show()

# Training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

sleep(0.5)
plt.show()

'''
Export the model

In the code above, you applied the TextVectorization layer to the dataset before feeding text to the model. 
If you want to make your model capable of processing raw strings (for example, to simplify deploying it), 
you can include the TextVectorization layer inside your model. 
To do so, you can create a new model using the weights you just trained.
'''

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
# This shows that raw text gets vectorized and the results are the same
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# To get predictions for new examples, you can simply call model.predict()
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

print(export_model.predict(examples))

