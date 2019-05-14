import logging
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
from .model_builder import ModelBuilder
from .doc2vec_builder import doc2VecBuilder
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class classifierBuilder(ModelBuilder):
    def __init__(self):
        super.__init__()

    def initialize_model(self):
        self.model = LogisticRegression()

    def train_model(self, d2v, training_vectors, training_labels):
        logging.info("Classifier training")
        train_vectors = doc2VecBuilder.get_vectors(d2v, len(training_vectors), 300, 'Train')
        self.model.fit(train_vectors, np.array(training_labels))
        training_predictions = self.model.predict(train_vectors)
        logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
        logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
        logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))


    def save_model(self, filename):
        joblib.dump(self.model,"./classifiers/"+ filename)

    def load_model(self,filename):
        if (os.path.isfile('./classifiers/' + filename)):
            loaded_model = joblib.load(filename)
            self.model = loaded_model
        else:
            self.model = None