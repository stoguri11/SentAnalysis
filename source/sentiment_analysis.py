import numpy as np
from numpy import array
from numpy import argmax
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Masking, Bidirectional
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def prepare_data(vecs, corpus, sentiments):
    '''
    build new list from the previously obtained GloVe embeddings and the "corpus"
    such that each word in every line of the "corpus" is replaced with its corresponding
    GloVe vector from "vecs". 
    Append the sentiment polarities to their corresponding tweets such that every element 
    of corpus now resembles [(polarity of tweet, [list of word vectors])]
    '''
    # assemble tweets vectors
    print('Assembling tweet vectors...\n')
    for i, line in enumerate(corpus):
        corpus[i] = line.strip().split()
        for j, word in enumerate(corpus[i]):
            for vec in vecs:
                if (corpus[i][j] == vec[0]):
                    corpus[i][j] = vec[1]

    # set tweet matrix length to longest tweet
    tweet_mat_len = max([len(line) for line in corpus])

    # pad tweet vectors to same length as tweet_mat_len
    print("padding tweet vectors...\n")
    temp1 = []
    for line in corpus:
        while len(line) < 45:
            pad = np.zeros(100, dtype=np.float64)
            line.append(pad)
        temp1.append(line)

    # sentiemnt vector will be one hot encoded vector 
    print('encoding sentiment labels...\n')
    temp2 = []
    for i, line in enumerate(sentiments):
        if (sentiments[i].strip() == "positive"):
            temp2.append(np.array([np.array([0,0,1]), temp1[i]]))       # [0,0,1] = Positive tweet
        elif ((sentiments[i].strip() == "neutral")):
            temp2.append(np.array([np.array([0,1,0]), temp1[i]]))       # [0,1,0] = Neutral tweet
        else:
            temp2.append(np.array([np.array([1,0,0]), temp1[i]]))       # [1,0,0] = Negative tweet   

    # Split tweet_vecs in 60% train, 20% test and 20% validation
    train, test, validate = np.split(temp2, [int(.6*len(temp2)), int(.8*len(temp2))])

    # build 2 numpy arrays for each of training, test and validate tweets 
    # and for their corresponding sentiments
    print("create train test and validate sets...\n")
    train_tweets = []
    train_sentiments = []
    for line in train:
        np_vec = []
        np_sent = []
        for vec in line[1]:
            np_vec.append(np.asarray(vec, dtype=np.float64))
        for sentiment in line[0]:
            np_sent.append(np.asarray(sentiment, dtype=np.float64))
        train_tweets.append(np.asarray(np_vec, dtype=np.float64))
        train_sentiments.append(np.asarray(np_sent, dtype=np.float64))
    train_tweets = np.asarray(train_tweets, dtype=np.float64)
    train_sentiments = np.asarray(train_sentiments, dtype=np.float64)
    
    test_tweets = []
    test_sentiments = []
    for line in test:
        np_vec = []
        np_sent = []
        for vec in line[1]:
            np_vec.append(np.asarray(vec, dtype=np.float64))
        for sentiment in line[0]:
            np_sent.append(np.asarray(sentiment, dtype=np.float64))
        test_tweets.append(np.asarray(np_vec, dtype=np.float64))
        test_sentiments.append(np.asarray(np_sent, dtype=np.float64))
    test_tweets = np.asarray(test_tweets, dtype=np.float64)
    test_sentiments = np.asarray(test_sentiments, dtype=np.float64)

    validate_tweets = []
    validate_sentiments = []
    for line in validate:
        np_vec = []
        np_sent = []
        for vec in line[1]:
            np_vec.append(np.asarray(vec, dtype=np.float64))
        for sentiment in line[0]:
            np_sent.append(np.asarray(sentiment, dtype=np.float64))
        validate_tweets.append(np.asarray(np_vec, dtype=np.float64))
        validate_sentiments.append(np.asarray(np_sent, dtype=np.float64))
    validate_tweets = np.asarray(validate_tweets, dtype=np.float64)
    validate_sentiments = np.asarray(validate_sentiments, dtype=np.float64)

    # calculate an approximate optimal number of hidden nodes for the network
    # 50 is the length of each word vector
    num_hidden_nodes = int(2/3 * ((tweet_mat_len/1.5) * 100))
    print(f"The number of hidden nodes is {num_hidden_nodes}.")

    # number of output labels (negative, neutral, positive)
    output_labels=3

    # return list with a tuple that contains all necessary data for LSTM
    return [(train_tweets, train_sentiments,
             test_tweets, test_sentiments,
             validate_tweets, validate_sentiments,
             tweet_mat_len, num_hidden_nodes, output_labels)]

def build_train_model(data):
    '''
    Using Keras deep learning library I am able to quickly set paramaters 
    and build an RNN with LSTM cells.
    batch_size will be the number of training samples processed before each update
    train_x is the array of tweet vectors
    train_y is the sentiment values
    '''
    for (train_tweets, train_sentiments, test_tweets, test_sentiments, validate_tweets, 
    validate_sentiments, tweet_mat_len, num_hidden_nodes, output_labels) in data:

        # number of data points fed into model per update
        batch_size=100

        # the dropout factor to help reduce overfitting
        drop_rate1 = 0.5
        drop_rate2 = 0.3

        # weight decay for vectors 
        w_decay = 1E-8

        # learning rate for Adam optimization
        LR = 0.005

        # Build the model
        print('Build model...')
        print(train_tweets.shape)
        model = Sequential()
        model.add(Masking(mask_value=np.zeros(100, dtype=np.float64), input_shape=(tweet_mat_len, 100)))

        model.add(LSTM(num_hidden_nodes, return_sequences=True, kernel_regularizer=l1_l2(w_decay), recurrent_regularizer=l1_l2(w_decay), bias_regularizer=l1_l2(w_decay), recurrent_dropout=(drop_rate1)))
        model.add(Dropout(drop_rate1))
        model.add(LSTM(int(num_hidden_nodes*1.4), return_sequences=False, kernel_regularizer=l1_l2(w_decay), recurrent_regularizer=l1_l2(w_decay), bias_regularizer=l1_l2(w_decay), recurrent_dropout=(drop_rate2)))
        model.add(Dropout(drop_rate2))

        model.add(Dense(units=output_labels))
        model.add(Activation('softmax'))
        
        optimizer = Adam(learning_rate=LR, clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())


        print("Training model...")
        training = model.fit(train_tweets, train_sentiments, batch_size=batch_size, epochs=20, verbose=1, validation_data=(validate_tweets, validate_sentiments))
        
        test_loss, test_acc = model.evaluate(test_tweets, test_sentiments)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))

        predictions = model.predict(test_tweets)
        print(predictions)



        def plot_graphs(history, metric):
            plt.plot(history.history[metric])
            plt.plot(history.history['val_'+metric], '')
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend([metric, 'val_'+metric])
            plt.show()

        plot_graphs(training, 'accuracy')
        plot_graphs(training, 'loss')



        return [training]