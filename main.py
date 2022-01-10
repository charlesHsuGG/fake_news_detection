import os
import time
import six
import itertools

from collections import Counter
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import getEmbeddings

print(tf.__version__)
print("Is there a GPU available: ")
print(tf.test.is_gpu_available())

units = 100
top_words = 5000
epoch_num = 5
batch_size = 64
embedding_dim = 32


class SequenceModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(SequenceModel, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units, dropout = 0.2, recurrent_dropout=0.2))
        self.sigmoid = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        gru_output = self.biLSTM(x)
        output = self.sigmoid(gru_output)
        return output


def loss_fn(inputs, targets):
    return tf.keras.losses.binary_crossentropy(targets, inputs)

def accuracy_fn(inputs, targets):
    return tf.keras.metrics.binary_accuracy(targets, tf.cast(inputs, tf.int64))
          

# @tf.function
def train_step(sequence, optimizer, inp, targs):
    with tf.GradientTape() as tape:
        labels = tf.expand_dims(targs, 1)

        outputs = sequence(inp)

        batch_loss = loss_fn(outputs, labels)

    batch_accuracy = accuracy_fn(outputs, targs)

    variables = sequence.variables

    gradients = tape.gradient(batch_loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_accuracy, batch_loss


if __name__ == "__main__":
    if not os.path.isfile('./xtr_shuffled.npy') or \
    not os.path.isfile('./xte_shuffled.npy') or \
    not os.path.isfile('./ytr_shuffled.npy') or \
    not os.path.isfile('./yte_shuffled.npy'):
       getEmbeddings.clean_data()

    xtr = np.load('./xtr_shuffled.npy')
    xte = np.load('./xte_shuffled.npy')
    y_train = np.load('./ytr_shuffled.npy')
    y_test = np.load('./yte_shuffled.npy')

    cnt = Counter()
    x_train = []
    for x in xtr:
        x_train.append(x.split())
        for word in x_train[-1]:
            cnt[word] += 1  
    
    most_common = cnt.most_common(top_words + 1)
    word_bank = {}
    id_num = 1
    for word, freq in most_common:
        word_bank[word] = id_num
        id_num += 1

    # Encode the sentences
    for news in x_train:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    y_train = list(y_train)

    # delete short news detect that isnt work
    index = 0
    while index < len(x_train):
        if len(x_train[index]) > 10:
            index += 1
        else:
            del x_train[index]
            del y_train[index]

    # generate test set
    y_test = list(y_test)
    x_test = []
    for x in xte:
        x_test.append(x.split())

    # Encode the sentences
    for news in x_test:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    max_review_length = 500
    X_train = pad_sequences(x_train, maxlen=max_review_length)
    X_test = pad_sequences(x_test, maxlen=max_review_length)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    steps_per_epoch = len(X_train) // batch_size

    for epoch in range(epoch_num):
        start = time.time()

        sequence = SequenceModel(top_words+2, embedding_dim, units, batch_size)
        optimizer = tf.keras.optimizers.Adam()

        total_loss = 0
        total_accuracy = 0
        index = 0

        
        for batch in range(0, steps_per_epoch):
            inps = X_train[index:index+batch_size]
            targs = y_train[index:index+batch_size]
            if len(inps) < batch_size:
                inps = np.hstack(inps,X_train[0:batch_size-len(inps)])
                targs = np.hstack(targs,y_train[0:batch_size-len(inps)])
            batch_accuracy, batch_loss = train_step(sequence, optimizer, inps, targs)
            loss = (sum(batch_loss.numpy()) / int(targs.shape[0]))
            accuracy = batch_accuracy.numpy().mean()
            total_loss += loss
            total_accuracy += accuracy

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            loss,
                                                            accuracy))
            index += batch_size + 1
        # saving (checkpoint) the model every 2 epochs
        # if (epoch + 1) % 2 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch,
                                            total_accuracy / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))