import pandas as pd
from sklearn.model_selection import train_test_split
from models.doc2vec_builder import doc2VecBuilder
from models.classifier_builder import classifierBuilder

class TextClassifier():

    def __init__(self):
        super().__init__()
        self.d2v = doc2VecBuilder()
        self.classifier = classifierBuilder()
        self.dataset = None

    def read_data(self,path):
        self.dataset = pd.read_csv(path, header=0, delimiter="\t")

    def prepare_all_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.dataset.review, self.dataset.sentiment, random_state=0,
                                                            test_size=0.1)
        x_train = doc2VecBuilder.label_sentences(x_train, 'Train')
        x_test = doc2VecBuilder.label_sentences(x_test, 'Test')
        all_data = x_train + x_test
        return x_train, x_test, y_train, y_test, all_data

    #def prepare_test_data(self):
    #    x_test = doc2VecBuilder.label_sentences(self.dataset.review, 'Test')
    #    y_test = self.dataset.sentiment
    #    return x_test, y_test

    def train_classifier(self, d2v_file, classifier_file):
        x_train, x_test, y_train, y_test, all_data = self.prepare_all_data()
        self.d2v.initialize_model(all_data)
        self.d2v.train_model()
        self.d2v.save_model(d2v_file)
        self.classifier.initialize_model()
        self.classifier.train_model(self.d2v, x_train, y_train)
        self.classifier.save_model(classifier_file)
        self.classifier.test_model(self.d2v, x_test, y_test)

    def test_classifier(self, d2v_file, classifier_file):
        #x_test, y_test = self.prepare_test_data()
        x_train, x_test, y_train, y_test, all_data = self.prepare_all_data()
        self.d2v.load_model(d2v_file)
        self.classifier.load_model(classifier_file)
        self.classifier.test_model(self.d2v, x_test, y_test)

if __name__ == "__main__":
    tc = TextClassifier()
    tc.read_data('./data/dataset.csv')
    tc.train_classifier("d2v.model", "joblib_model.pkl")
    #tc.test_classifier("d2v.model","joblib_model.pkl")

