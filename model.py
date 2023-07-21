
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense
from keras.metrics import Precision, Recall
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping



def load_glove_embedding_matrix(path, word_index, embed_dim):
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def bilstm(vocabulary_size,seq_len,drop,hidden_cells,n_classes,embed_dim=100,use_glove=False, glove_matrix=None):
   model = Sequential()
   if use_glove:
        model.add(Embedding(vocabulary_size, embed_dim, 
                            weights=[glove_matrix], input_length=seq_len,
                             trainable=True))
   else:
        model.add(Embedding(vocabulary_size, embed_dim, input_length=seq_len
                            ))
   model.add(Dropout(drop))
   model.add(Bidirectional(LSTM(hidden_cells, return_sequences=True, 
                                 dropout=drop)))
   model.add(Dense(n_classes, activation='softmax'))
   model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy',
                           Precision(),
                           Recall()])
   model.summary()
   return model