import logging
import random
import numpy as np
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec

from .model_builder import ModelBuilder
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class doc2VecBuilder(ModelBuilder):

    def __init__(self, corpus):
        self.corpus = corpus
