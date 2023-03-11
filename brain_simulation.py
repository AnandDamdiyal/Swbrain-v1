import numpy as np
from models.neural_networks import NeuralNetwork
from models.markov_models import MarkovModel
from models.hidden_markov_models import HiddenMarkovModel


class BrainSimulation:
    """
    Class for simulating brain activity and performing classification using different models.
    """
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        """
        Initializes the BrainSimulation class with the specified parameters.

        Args:
        - n_channels (int): Number of EEG channels.
        - n_samples (int): Number of time samples for each EEG signal.
        - n_classes (int): Number of classes for classification.
        """
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def generate_data(self, n_train: int, n_test: int):
        """
        Generates synthetic EEG data for training and testing.

        Args:
        - n_train (int): Number of training samples to generate.
        - n_test (int): Number of test samples to generate.
        """
        self.X_train = np.random.randn(n_train, self.n_channels, self.n_samples)
        self.y_train = np.random.randint(self.n_classes, size=n_train)
        self.X_test = np.random.randn(n_test, self.n_channels, self.n_samples)
        self.y_test = np.random.randint(self.n_classes, size=n_test)
    
    def classify_nn(self, hidden_layers: list, activations: list, optimizer: str, epochs: int, batch_size: int):
        """
        Trains and evaluates a neural network classifier on the generated data.

        Args:
        - hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
        - activations (list): List of activation functions for each hidden layer.
        - optimizer (str): Optimization algorithm for training the neural network.
        - epochs (int): Number of epochs to train the neural network for.
        - batch_size (int): Size of the minibatches for stochastic gradient descent.
        """
        nn = NeuralNetwork(hidden_layers, activations, optimizer)
        nn.train(self.X_train, self.y_train, epochs, batch_size)
        y_pred = nn.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)
        return accuracy
    
    def classify_mm(self, order: int):
        """
        Trains and evaluates a Markov Model classifier on the generated data.

        Args:
        - order (int): Order of the Markov Model.

        Returns:
        - accuracy (float): Classification accuracy of the Markov Model.
        """
        mm = MarkovModel(order)
        mm.train(self.X_train, self.y_train)
        y_pred = mm.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)
        return accuracy
    
    def classify_hmm(self, n_states: int, n_components: int, covariance_type: str, n_iter: int):
        """
        Trains and evaluates a Hidden Markov Model classifier on the generated data.

        Args:
        - n_states (int): Number of hidden states in the HMM.
        - n_components (int): Number of Gaussian mixture components in each state.
        - covariance_type (str): Covariance type for the Gaussian mixture models.
        - n_iter (int): Number of iterations for training the HMM.

        Returns:
        - accuracy (float): Classification accuracy of the Hidden Markov Model.
        """
        # predict using the fitted HMM model
    predicted_states = model.predict(X_test)
    
    # calculate classification accuracy
    accuracy = accuracy_score(y_test, predicted_states)
    
    # return the classification accuracy
    return accuracy
   
