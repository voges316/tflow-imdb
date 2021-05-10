# TensorFlow IMDB Work

Example scripts to train TensorFlow 2 Neural Network for text sentiment analysis using IMDB dataset.

## Prerequisites

- Linux
- Python 3.8, tensorflow-cpu, pandas, numpy, nltk, beautifulsoup4

## Running

Run the simpler imdb_binary_classification.py, or the more imdb_binary_class_cnn.py, which will parse and read the 
imdb movie review dataset, and create models as well as print their evaluation metrics. Removing stop words does not
appear to increase the accuracy of imdb_binary_class, which uses a straightforward stack of neural nets.

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

You can run the more complex imdb_binary_class_cnn.py, but the accuracy appears to be less. However, normalization does 
appear to improve accuracy when using LSTM's. Removing stop word using imdb_parser.py appeared to increase results 
from approx 83% to 85%

```
# Use the imdb_parser to preprocess the raw data files. This will alter the text in each file.

./imdb_parser.py --dir aclImdb/train/pos
./imdb_parser.py --dir aclImdb/train/neg
./imdb_parser.py --dir aclImdb/test/pos
./imdb_parser.py --dir aclImdb/test/neg

./imdb_binary_class_cnn.py

Train Accuracy: 0.919
Test Accuracy: 0.872
```

Finally, running imdb_bert.py will download and train the bert wiki experts model using the imdb dataset. Beware that
it is resource intensive, taking approximately 1.5 hours per epoch. 
Removing stop words 'increased' test accuracy from 0.8902 to 0.8903, which is likely insignificant.

```

./imdb_bert.py

...
Epoch 5/5
625/625 [==============================] - 6091s 10s/step - loss: 0.0881 - binary_accuracy: 0.9733 - val_loss: 0.5012 - val_binary_accuracy: 0.8952
Loss: 0.5062994360923767
Accuracy: 0.8903200030326843

```

## Built With

* [Python3.8](https://www.python.org/) - The scripting code used
* [TensorFlow](https://www.tensorflow.org/) - For creating neural networks
* [NLTK](https://www.nltk.org/) - Natural Language Toolkit
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Easy HTML/XML parsing
* [Pandas](https://pandas.pydata.org/) - Data Analysis Library

