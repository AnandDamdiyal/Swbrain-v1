import numpy as np
from hmmlearn import hmm


class HiddenMarkovModel:
    """
    Class for training and using Hidden Markov Models (HMMs) for EEG signal classification.
    """
    def __init__(self, n_states: int, n_components: int, covariance_type: str, n_iter: int):
        """
        Initializes the HiddenMarkovModel class with the specified parameters.

        Args:
        - n_states (int): Number of hidden states in the HMM.
        - n_components (int): Number of Gaussian mixture components in each state.
        - covariance_type (str): Covariance type for the Gaussian mixture models.
        - n_iter (int): Number of iterations for training the HMM.
        """
        self.n_states = n_states
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.hmm = None
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the HMM on the specified training data and labels.

        Args:
        - X (np.ndarray): Array of shape (n_samples, n_features) containing the training data.
        - y (np.ndarray): Array of shape (n_samples,) containing the labels corresponding to the training data.
        """
        # Convert labels to integers
        label_map = {label: i for i, label in enumerate(np.unique(y))}
        y_int = np.array([label_map[label] for label in y])

        # Initialize and fit the HMM
        self.hmm = hmm.GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type, n_iter=self.n_iter)
        self.hmm.fit(X, y_int)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test data.

        Args:
        - X (np.ndarray): Array of shape (n_samples, n_features) containing the test data.

        Returns:
        - y_pred (np.ndarray): Array of shape (n_samples,) containing the predicted labels.
        """
        # Predict the most likely sequence of hidden states for each test sample
        state_sequence = self.hmm.predict(X)

        # Convert the state sequence to class labels
        label_map = {i: label for label, i in label_map.items()}
        y_pred = np.array([label_map[state] for state in state_sequence])

        return y_pred
