from abc import  ABC, abstractmethod

class ModelBuilder(ABC):

    def __init__(self):
        self.model = None
        super().__init__()

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass