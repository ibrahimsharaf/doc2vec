from .model import Model
from .doc2vec_model import doc2VecModel

import logging
import os
import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')


class classifierModel(Model):
    def __init__(self):
        super().__init__()

    def initialize_model(self):
        self.model = LogisticRegression()

    def train_model(self, d2v, training_vectors, training_labels):
        logging.info("Classifier training")
        train_vectors = doc2VecModel.get_vectors(
            d2v, len(training_vectors), 300, 'Train')
        self.model.fit(train_vectors, np.array(training_labels))
        training_predictions = self.model.predict(train_vectors)
        logging.info(
            'Training predicted classes: {}'.format(np.unique(
                training_predictions)))
        logging.info(
            'Training accuracy: {}'.format(
                accuracy_score(training_labels, training_predictions)))
        logging.info(
            'Training F1 score: {}'.format(
                f1_score(
                    training_labels, training_predictions,
                    average='weighted')))

    def test_model(self, d2v, testing_vectors, testing_labels):
        logging.info("Classifier testing")
        test_vectors = doc2VecModel.get_vectors(
            d2v, len(testing_vectors), 300, 'Test')
        testing_predictions = self.model.predict(test_vectors)
        logging.info(
            'Testing predicted classes: {}'.format(
                np.unique(testing_predictions)))
        logging.info(
            'Testing accuracy: {}'.format(
                accuracy_score(testing_labels, testing_predictions)))
        logging.info(
            'Testing F1 score: {}'.format(
                f1_score(
                    testing_labels, testing_predictions,
                    average='weighted')))

    def predict(self, d2v, testing_vectors):
        logging.info("Classifier Predicting")
        test_vectors = doc2VecModel.get_vectors(
            d2v, len(testing_vectors), 300, 'Test')
        testing_predictions = self.model.predict(test_vectors)
        logging.info(testing_predictions)
