from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def bow(tweets, sentiments):        
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    train_data_features = vectorizer.fit_transform(tweets)

    vecs =[]
    for line in tweets:
        doc = [line]
        vec = vectorizer.transform(doc).toarray()
        vecs.append(vec)
    
    np_vecs = np.asarray(vecs, dtype=np.float64)

    sents = []
    for i, line in enumerate(sentiments):
        if (sentiments[i].strip() == "positive"):
            sents.append(np.array([0,0,1]))       # [0,0,1] = Positive tweet
        elif ((sentiments[i].strip() == "neutral")):
            sents.append(np.array([0,1,0]))       # [0,1,0] = Neutral tweet
        else:
            sents.append(np.array([1,0,0]))
    
    np_sents = np.asarray(sents, dtype=np.float64)

    train_tweets = np_vecs[:int(len(np_vecs)*0.6)]
    test_tweets = np_vecs[int(len(np_vecs)*0.6):int(len(np_vecs)*0.8)]
    validate_tweets = np_vecs[int(len(np_vecs)*0.8):]
    train_sentiments = np_sents[:int(len(np_sents)*0.6)]
    test_sentiments = np_sents[int(len(np_sents)*0.6):int(len(np_sents)*0.8)]
    validate_sentiments = np_sents[int(len(np_sents)*0.8):]

    tweet_mat_len = 5000    # all bow vectors are 500 long  

    # calculate an approximate optimal number of hidden nodes for the network
    # 50 is the length of each word vector
    num_hidden_nodes = 500
    print(f"The number of hidden nodes is {num_hidden_nodes}.")

    for line in train_tweets:
        print(line.size)

    # number of output labels (negative, neutral, positive)
    output_labels=3

    # return list with a tuple that contains all necessary data for LSTM
    return [(train_tweets, train_sentiments,
             test_tweets, test_sentiments,
             validate_tweets, validate_sentiments,
             tweet_mat_len, num_hidden_nodes, output_labels)]

    return data;