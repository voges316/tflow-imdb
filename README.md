# TensorFlow IMDB Work

Example scripts to train TensorFlow 2 Neural Network for text sentiment analysis using IMDB dataset.

## Prerequisites

- Linux
- Python 3.8, tensorflow-cpu, pandas, numpy, nltk, beautifulsoup4

## Running

Run the simpler imdb_binary_classification.py, or the more imdb_binary_class_cnn.py, which will parse and read the 
imdb movie review dataset, and create models as well as print their evaluation metrics.

```

# download dataset
cd src; ./download_imdb.sh

# Run neural network on raw dataset
./imdb_binary_classification.py

...
Loss & accuracy on train dataset
Loss:  0.16128292679786682
Accuracy:  0.9474499821662903

Loss & accuracy on test dataset
Loss:  0.2768723964691162
Accuracy:  0.8918799757957458

```

You can run the more complex imdb_binary_class_cnn.py, but the accuracy appears to be less, however it is affected by 
word normalization more. Removing stop words appears to increase results from approx 83% to 85%

```
# Use the imdb_parser to preprocess the raw data files. This will alter the text in each file.

./imdb_parser.py --dir aclImdb/train/pos
./imdb_parser.py --dir aclImdb/train/neg
./imdb_parser.py --dir aclImdb/test/pos
./imdb_parser.py --dir aclImdb/test/neg

./imdb_binary_class_cnn.py

```

## Built With

* [Python3.8](https://www.python.org/) - The scripting code used
* [TensorFlow](https://www.tensorflow.org/) - For creating neural networks
* [NLTK](https://www.nltk.org/) - Natural Language Toolkit
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Easy HTML/XML parsing
* [Pandas](https://pandas.pydata.org/) - Data Analysis Library

