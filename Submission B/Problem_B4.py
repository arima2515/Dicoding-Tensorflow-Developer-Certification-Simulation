# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    sentences = bbc['text'].to_numpy()
    labels = bbc['category'].to_numpy()

    split = int(len(sentences) * training_portion)

    training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(sentences, labels,
                                                                                              train_size=training_portion,
                                                                                              random_state=42)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    label_tokenizer = Tokenizer()

    label_tokenizer.fit_on_texts(labels)
    tokenizer.fit_on_texts(training_sentences)

    sequences = tokenizer.texts_to_sequences(training_sentences)

    train_label_sequences = label_tokenizer.texts_to_sequences(training_labels)
    train_label_sequences = np.array(train_label_sequences) - 1
    test_label_sequences = label_tokenizer.texts_to_sequences(testing_labels)
    test_label_sequences = np.array(test_label_sequences) - 1

    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(padded, train_label_sequences, epochs=20, validation_data=(testing_padded, test_label_sequences))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
