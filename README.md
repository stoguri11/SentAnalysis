# Sentiment Analysis Application

Twitter Sentiment analysis with GloVe embeddings and LSTM. 

REQUIREMENTS
1) Python v3.7
2) pipenv - install with 'pip install pipenv' in the project directory



To run the python code for training and making predictions follow these instruction:

1) cd to the project directory.

2) In the terminal run 'pipenv install' - this will install dependiencies from pipfile and create pipfile.lock

3) run 'pipenv shell'

4) to make predictions place your file in the './data' directory and call it 'predictions.csv'. the file only needs to be formatted exactly like the data in 'training.csv'
   you can use a subset of training.csv to test the functionality. 
    4.1) run the command 'pipenv run python source/predict.py' - predictions will be made and output to the terminal

5) to train the model run the command 'pipenv run python source/training.py' 



