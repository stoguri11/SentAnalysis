#!/usr/bin/env python
import csv
from collections import Counter
from math import log
from random import shuffle
import numpy as np
from scipy import sparse


def get_vocab(corpus):
    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    '''
    Build a word co-occurrence matrix.

    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form (i_main, i_context, cooccurrence).
    Where: 
    `i_main` is the ID of the main word in the cooccurrence, 
    `i_context` is the ID of the context word, 
    `cooccurrence` is the `X_{ij}` cooccurrence value as described in (Pennington et al. 2014).
    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    '''

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    # Collect cooccurrences internally as a sparse matrix
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

    for i, line in enumerate(corpus):
        if (i%1000==0):
            print("Building cooccurrence matrix on line: ", i)
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Weight vectors by inverse of distance between center and
                # context word to represent strength of relationship
                distance = contexts_len - left_i
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    # if the coccurrence value is lower than min_count i.e. these two words do 
    # often enough together, then they are ignored/
    for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][0] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][0] < min_count:
                continue
            
            #print(i, " ", j, " ", data[data_idx])
            yield i, j, data[data_idx]


def run_iter(vocab, data, learning_rate=0.07, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; 
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (w_main, w_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = ((w_i . w_j) + b_i + b_j - log(X_{ij})) $$
        cost_inner = (w_main.dot(w_context) + b_main[0] + b_context[0] - log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        grad_main = cost_inner * w_context
        grad_context = cost_inner * w_main

        # Compute gradients for bias terms
        grad_bias_main = cost_inner
        grad_bias_context = cost_inner

        # perform adaptive updates
        w_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        w_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, iterations=50, **kwargs):

    vocab_size = len(vocab)
    vector_size=100

    # W - Word vector matrix. This matrix is (2V) * d, where V is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. There are two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # Bias terms, for each vector. An array of size
    # 2V, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training will be done using adaptive gradient descent (AdaGrad) algorithm,
    # therefore sums of all previous squared gradients must be stored.
    #
    # initialise all values to 1 so initial adaptive learning rate is 
    # equal to global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size), dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # Build a reusable list from the given cooccurrence generator,
    # pre-fetching all necessary data.
    data = [(W[i_main], 
             W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main],
             gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]


    for i in range(iterations):
        print("\tBeginning iteration of training ", i+1, "...")

        cost = run_iter(vocab, data, **kwargs)

        print("\t\tDone (cost = ", cost,")")

    return W

def GloVeCtrl(corpus):

    print("Creating vocab dictionary..")
    vocab = get_vocab(corpus)
    print("Completed\n")

    print("Building cooccurrence matrix...")
    cooccurrences = build_cooccur(vocab, corpus)
    print("Completed\n")

    W = train_glove(vocab, cooccurrences)
    print("GloVe embeddings trained, number of vectors = ", len(W)/2)

    # Normalize word vectors
    for i, row in enumerate(W):
        W[i, : ] /= np.linalg.norm(row)
    
    # obtain mean vector from context and main vectors for each word
    main_vecs = np.array(W[:len(vocab), :])
    context_vecs = np.array(W[len(vocab):, :])
    mean_vecs = []
    for i, vec in enumerate(main_vecs):
        mean_vec = []
        for j, val in enumerate(vec):
            mean_val = np.mean([val, context_vecs[i][j]])
            mean_vec.append(mean_val)
        mean_vecs.append(np.asarray(mean_vec, dtype=np.float64))
    W = np.asarray(mean_vecs, dtype=np.float64)

    id2word = dict((id, word) for word, (id, _) in vocab.items())
    W = W.tolist()

    # append word vectors to their actual word
    # E.g. [("happy", [vector for "happy"])]
    words_and_vecs = []
    for i, vector in enumerate(W):
        words_and_vecs.append([id2word[i], vector])
    
    return np.asarray(words_and_vecs), len(vocab)

