import tensorflow as tf

def make_predictions(data):
    trained_model = tf.keras.models.load_model('./data/trained_model.h5')

    for (train_tweets, train_sentiments, test_tweets, test_sentiments, validate_tweets, 
    validate_sentiments, tweet_mat_len, num_hidden_nodes, output_labels) in data:
        # Check its architecture
        trained_model.summary()
        predictions = trained_model.predict(train_tweets)
        print(predictions)