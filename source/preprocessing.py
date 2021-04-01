import csv
import re
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

stops = []
with open("./data/stop_words.txt", encoding='utf-8') as stop_words:
    lines = csv.reader(stop_words)
    for line in lines:
        stops.append(line[0])

# remove stopwords, punctuation, repeated characters, URL's 
def remove_noise(data):



    # array to hold processed data
    no_stops = []
    for row in data:
        txt = row[3]

        # result = regex to remove @'s, #'s, Usernames and non-alphanumeric characters
        # result_words = list comprehension that filters out stop words
        no_urls = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(/#\w+\s*/)", " ", txt).split())
        no_nums = ''.join([i.lower() for i in no_urls if not i.isdigit()])
        result_words = [word for word in no_nums.split() if word not in stops]

        # remove characters after 4 or more consecutive duplicates, calling remove_duplicates function
        for i in range(len(result_words)):
            if len(result_words[i]) > 3 :
                result_words[i] = ''.join(remove_duplicates(result_words[i].lower()))
            else:
                result_words[i] = result_words[i]

        row[3] = result_words
        if len(result_words) > 0:
            no_stops.append(row)

    return no_stops


# A function that removes consecutive duplicates greater than 3 from string S
def remove_duplicates(S): 
    
    # regular expression to split S at each unique character.
    # E.g. 'Juuuuusssssttttt' becomes ['J', 'uuuuu', 'sssss', 'ttttttt'].
    duplicate_list = [m.group(0) for m in re.finditer(r"(\w)\1*", S)]

    # ['J', 'uuuuu', 'sssss', 'ttttttt'] becomes [juuusssttt]
    for i in range(len(duplicate_list)):
        if len(duplicate_list[i]) > 3:
            duplicate_list[i] = duplicate_list[i][0:3]

    S = duplicate_list
    
    return S 


# perform part-of-speech tagging in preparation for lemmatization
def pos_tagging(data):

    for line in data:
        for word in line[3]:
            if word in stops:
                line[3].remove(word)
            

    parts_of_speech = [nltk.pos_tag(lines[3]) for lines in data]

    # nltk pos tagger returns treebank pos tags, 
    # this converts into wordnet pos tags for the lemmatizer
    for i in range(len(parts_of_speech)):
        line = parts_of_speech[i]
        for j in range(len(line)):
            pair = line[j]
            wordnet_pos = get_wordnet_pos(pair[1])
            if len(wordnet_pos) > 0:
                line[j] = (pair[0], wordnet_pos)
            else:
                line[j] = pair[0]

    for i in range(len(data)):
        line = data[i]
        line[3] = parts_of_speech[i]

    return data


# turn treebank pos tags 'NN', 'JJ', etc... into wordnet pos tags 'n', 'a', etc...
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return "n"


# lemmatize the tweets using wordnet lemmatizer
def lemmatisation(data):
    lemmatizer = WordNetLemmatizer()
    
    for line in data:
        txt = line[3]
        for i in range(len(txt)):
            txt[i] = lemmatizer.lemmatize(txt[i][0], pos=txt[i][1])


    return data

