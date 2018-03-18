import logging
from random import shuffle

import gensim
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

LabeledSentence = gensim.models.doc2vec.LabeledSentence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
train_path = './Training.tsv'
test_path = './Testing.tsv'
training = pd.read_table(train_path , header= None , names = ['entry' , 'label'])
testing = pd.read_table(test_path, header= None, names = ['entry' , 'label'])

x_train = training.entry
y_train = training.label
x_test = testing.entry
y_test = testing.label

# some text processing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

# Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


allXs = x_train + x_test
x_train = labelizeReviews(x_train, 'Train')
x_test = labelizeReviews(x_test, 'Test')
allXs = labelizeReviews(allXs, 'All')

# Instantiate Doc2Vec model and build vocab
model = gensim.models.doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(allXs)
# model = doc2vec.Doc2Vec.load('doc2vec_model')
# Pass through the data set multiple times, shuffling the training reviews each time to improve accuracy
for epoch in range(20):
    model.train(utils.shuffle(x_train))

model.save('trained_model')

# Get training set vectors from our models
def getVecs(model, corpus, size, vecs_type):
    vecs = np.zeros((len(corpus), size))
    for i in range(0 , len(corpus)):
        index = i
        if vecs_type == 'Test':
            index = index + len(x_train)
        prefix = 'All_' + str(index)
        vecs[i] = model.docvecs[prefix]
    return vecs 

# Get train vectors
train_vecs = getVecs(model, x_train, 100, 'Train')
print train_vecs.shape

# Train model over test set
for epoch in range(20):
    model.train(utils.shuffle(x_test))

# Construct vectors for test reviews
test_vecs = getVecs(model, x_test, 100, 'Test')
print test_vecs.shape
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# Train classifier
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
lr.fit(train_vecs, y_train)
print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))
