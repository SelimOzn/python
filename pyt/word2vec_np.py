import string
import numpy as np
from corps import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
import random
from glob import glob
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
import sys

size = int(input("Enter the vocabulary size you want: "))
# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
  V = 20000
  files = glob('large_files/enwiki*.txt')
  all_word_counts = {}
  for f in files:
    for line in open(f, encoding="utf8"):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          for word in s:
            if word not in all_word_counts:
              all_word_counts[word] = 0
            all_word_counts[word] += 1
  print("finished counting")

  V = min(V, len(all_word_counts))
  all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

  top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
  word2idx = {w:i for i, w in enumerate(top_words)}
  unk = word2idx['<UNK>']

  sents = []
  for f in files:
    for line in open(f, encoding="utf8"):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          # if a word is not nearby another word, there won't be any context!
          # and hence nothing to train!
          sent = [word2idx[w] if w in word2idx else unk for w in s]
          sents.append(sent)
  return sents, word2idx

def get_context(position, window_size, sentence):
    context = []
    start_ind = max(0, (position-window_size))
    final_ind = min((position+window_size), len(sentence))
    for context_ind, context_word in enumerate(sentence[start_ind: final_ind], start=start_ind):
        if context_ind != position:
            context.append(context_word)
    return context


def sgd(inputs,targets, label, learning_rate,W , V):
    activation = W[inputs]
    prob = sigmoid(activation.dot(V[:, targets])) #prob kaç elemanlı??

    gradient_V = np.outer(W[inputs], (prob-label))
    gradient_W = np.sum((prob-label)*V[:, targets], axis=1)

    V[:, targets] -= learning_rate*gradient_V
    W[inputs] -= learning_rate*gradient_W

    cross_ent_loss = label * np.log(prob + (1e-10)) + (1 - label) * np.log((1-prob) + (1e-10)) #Kaç elemanlı??
    return cross_ent_loss.sum()


def train_model(learning_rate, final_lr, D, window_size, threshold, savedir):
    sentences, word2ind = get_wiki()
    vocabulary_size = len(word2ind)
    lr = learning_rate
    W = np.random.rand(vocabulary_size, D)
    V = np.random.rand(D, vocabulary_size)

    num_negatives = 5  # number of negative samples to draw per input word
    epochs = 20

    learning_rate_delta = (lr-final_lr) / epochs
    freq_dist = get_negative_samples(sentences, vocabulary_size)
    p_drop = drop_word(freq_dist, threshold, vocabulary_size)

    costs = []
    for epoch in range(epochs):

        counterx = 0

        t0 = datetime.now()
        cost = 0
        np.random.shuffle(sentences)
        for sentence in sentences:

            if (counterx % 1000) == 0:
                print(counterx)

            prob_sentence = p_drop[sentence]
            random_values = np.random.rand(len(sentence))
            drop_cond = random_values < prob_sentence
            new_sentence = np.delete(sentence, drop_cond)
            while True:
                if len(new_sentence) < 2:
                    random_values = np.random.rand(len(sentence))
                    drop_cond = random_values < prob_sentence
                    new_sentence = np.delete(sentence, drop_cond)
                else:
                    break


            random_position_choice = np.random.choice(len(new_sentence), size=len(new_sentence), replace=False)
            for pos in random_position_choice:

                word = new_sentence[pos]
                context_words = get_context(pos, window_size,new_sentence)
                targets = np.array(context_words)
                negative_samp_word = np.random.choice(vocabulary_size, p=freq_dist)

                train = sgd(inputs=word, targets=targets, label=1, learning_rate=lr, W=W, V=V)
                cost += train
                train_neg_samp = sgd(inputs=negative_samp_word, targets=targets, label=0, learning_rate=lr, W=W, V=V)
                cost += train_neg_samp
            counterx += 1

        costs.append(cost)
        lr -= learning_rate_delta

        dt = datetime.now() - t0
        print("epoch complete:", epoch, "cost:", cost, "process time:", dt)

    plt.plot(costs)
    plt.show()

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open("%s/word2ind.json" %savedir, "w") as f:
        json.dump(word2ind, f)

    np.savez("%s/weights" % savedir, W, V)

    return word2ind, W, V

def get_negative_samples(sentences, vocabulary_size):
    word_freq = np.zeros(vocabulary_size)

    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1

    word_freq = word_freq ** 0.75
    word_freq = word_freq / word_freq.sum()
    assert (np.all(word_freq > 0))
    return word_freq

def drop_word(freq_word_distrubition, threshold, vocab_size):

    drop_prob = np.zeros(vocab_size)
    drop_prob = 1 - np.sqrt((threshold/freq_word_distrubition))

    return drop_prob

def load_model(load_dir):

    with open("%s/word2ind.json" % load_dir) as f:
        word2ind = json.load(f)

    npz = np.load("%s/weights.npz" % load_dir)
    W = npz["arr_0"]
    V = npz["arr_1"]

    return word2ind, W, V


def find_analogy(positive1,negative1,positive2,negative2,word2ind,ind2word,W):
    V, D = W.shape

    print("Analogy: %s - %s = %s - %s" %(positive1, negative1, positive2, negative2))
    for word in (positive1, negative1, positive2, negative2):
        if word not in word2ind:
            print("Sorry, %s not in vocabulary" % word)
            return

    p1 = W[word2ind[positive1]]
    n1 = W[word2ind[negative1]]
    p2 = W[word2ind[positive2]]
    n2 = W[word2ind[negative2]]

    word_to_find = p1 - n1 + n2

    pairwise_distance_arr = pairwise_distances(word_to_find.reshape(1, D), W, metric="cosine").reshape(V)
    best_dist_inds = pairwise_distance_arr.argsort()[:10]

    best_ind = -1
    impos_inds = []
    for word in (positive1, negative1, negative2):
        impos_inds.append(word2ind[word])

    for best_word in best_dist_inds:
        if best_word not in impos_inds:
            best_ind = best_word
            break

    print("Analogy found by model: %s - %s = %s - %s" %(positive1, negative1,ind2word[best_ind], negative2))
    print("10 words closest to %s" %positive2)
    for inds in best_dist_inds:
        print(ind2word[inds])

def test_model(word2idx, idx2word, W, V):

    for w_e in (W, (W + V.T) / 2):
        print("Test analogy for %s" % w_e)

        find_analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, w_e)
        find_analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, w_e)
        find_analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, w_e)
        find_analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, w_e)
        find_analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'he', 'she', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, w_e)
        find_analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, w_e)
        find_analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, w_e)
        find_analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, w_e)
        find_analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, w_e)
        find_analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, w_e)
        find_analogy('february', 'january', 'december', 'november', word2idx, idx2word, w_e)
        find_analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, w_e)
        find_analogy('week', 'day', 'year', 'month', word2idx, idx2word, w_e)
        find_analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, w_e)
        find_analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, w_e)
        find_analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, w_e)
        find_analogy('france', 'french', 'england', 'english', word2idx, idx2word, w_e)
        find_analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, w_e)
        find_analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, w_e)
        find_analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, w_e)
        find_analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, w_e)
        find_analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, w_e)


if __name__ == '__main__':
    word2idx, W, V = train_model(threshold=1e-5, learning_rate=0.025, final_lr=0.0001, window_size=5, D=50, savedir="w2v_model")
    #word2idx, W, V = load_model('w2v_model')
    ind2word = {i: w for w, i in word2idx.items()}
    test_model(word2idx, ind2word, W, V)

