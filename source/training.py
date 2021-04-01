import csv
import re
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import remove_noise, pos_tagging, lemmatisation
from GloVe import GloVeCtrl
from sentiment_analysis import prepare_data, build_train_model
from BOW import bow

# loading dataset 
# dataset from http://help.sentiment140.com/for-students/
with open("data/training.csv", encoding='utf-8') as training_data:
    lines = csv.reader(training_data)
 
    # remove noise
    print("removing noise...\n")
    noise_removed = remove_noise(lines)
    
    # part-of-speech tagging and lemmatisation
    print("performing lemmatisation...\n")
    pos_tagged = pos_tagging(noise_removed)
    lemmatised = lemmatisation(pos_tagged)

    # seperate text and sentiments
    corpus = [" ".join(lemmas[3]) for lemmas in lemmatised]
    sentiments = [row[1] for row in lemmatised]

    # obtain GloVe embeddings for the corpus.
    # gets list of tuples in form [("word", [glove vector associated with word])]  
    # use comment to switch bettwen getting GloVe or BOW vectors
    glove_embeddings, V = GloVeCtrl(corpus)
    #bow_embeddings = bow(corpus, sentiments)
    print("glove length: ", len(glove_embeddings))

    # prepare data - 
    # comment below when using Bow embeddings, uncomment when using GloVe
    prepared_data = prepare_data(glove_embeddings, corpus, sentiments)
    
    # Train classifier
    trained = build_train_model(prepared_data)








    


