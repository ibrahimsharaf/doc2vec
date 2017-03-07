import gensim
import logging
import numpy as np
import pandas as pd
import numpy as np
from random import shuffle
from sklearn import utils
from gensim.models import doc2vec
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

LabeledSentence = gensim.models.doc2vec.LabeledSentence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
train_path = ./Training.tsv"
test_path = './Testing.tsv'
training = pd.read_table(train_path , header= None , names = ['entry' , 'label'])
testing = pd.read_table(test_path, header= None, names = ['entry' , 'label'])

# labels = [1,2,3,5,6]
# # 1 -> Title
# # 2 -> Price
# # 3 -> Manufacturer
# # 5 -> Category
# # 6 -> Stock Status
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

# # Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# # We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# # a dummy index of the review.
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
for i in range(10):
	print x_train[i]
  print x_test[i]

# Instantiate Doc2Vec model and build vocab
model = doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(allXs)
# model = doc2vec.Doc2Vec.load('doc2vec_model')
#Pass through the data set multiple times, shuffling the training reviews each time to improve accuracy
for epoch in range(20):
    model.train(utils.shuffle(x_train))

model.save('Model_after_train')
model = doc2vec.Doc2Vec.load('Model_after_train')
#print model.docvecs['All_0']


# get training set vectors from our models
def getVecs(model, corpus, size, vecs_type):
	vecs = np.zeros((len(corpus), size))
	for i in range(0 , len(corpus)):
		index = i
		if(vecs_type == 'Test'):
			index = index + 183891
		prefix = 'All_' + str(index)
		vecs[i] = model.docvecs[prefix]
	return vecs 

# get train vectors
train_vecs = getVecs(model, x_train, 100, 'Train')
print train_vecs.shape

# train model over test set
for epoch in range(20):
    model.train(utils.shuffle(x_test))

# Construct vectors for test reviews
test_vecs = getVecs(model, x_test, 100, 'Test')
print test_vecs.shape
model.save('Model_after_test')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# # train classifier
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
lr.fit(train_vecs, y_train)
# print model.docvecs['Train_0']
# print train_vecs[0]
# print y_train[0]
print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)
