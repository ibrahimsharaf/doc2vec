import logging
import re

import gensim
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import doc2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
LabeledSentence = gensim.models.doc2vec.LabeledSentence


def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    x_train, x_test, y_train, y_test = train_test_split(dataset.review, dataset.sentiment, random_state=0, test_size=0.10)
    data = x_train.tolist() + x_test.tolist()
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all = label_sentences(data, 'All')
    return x_train, x_test, y_train, y_test, all


def clean_text(text):
    # Remove HTML
    review_text = BeautifulSoup(text).get_text()
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Remove stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(LabeledSentence([v], [label]))
    return labeled


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        index = i
        if vectors_type == 'Test':
            index = index + len(x_train)
        prefix = 'All_' + str(index)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Building Doc2Vec model")
    d2v = doc2vec.Doc2Vec(min_count=1, window=3, vector_size=100, sample=1e-3, seed=1, workers=5)
    d2v.build_vocab(corpus)
    return d2v


def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Train Doc2Vec on training set")
    d2v.train(training_vectors, total_examples=len(training_vectors), epochs=d2v.iter)
    train_vectors = get_vectors(d2v, len(training_vectors), 100, 'Train')
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Train Doc2Vec on testing set")
    d2v.train(testing_vectors, total_examples=len(testing_vectors), epochs=d2v.iter)
    test_vectors = get_vectors(d2v, len(testing_vectors), 100, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all = read_dataset('dataset.csv')
    d2v_model = train_doc2vec(all)
    classifier = train_classifier(d2v_model, x_train, y_train)
    test_classifier(d2v_model, classifier, x_test, y_test)
