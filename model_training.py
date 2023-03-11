# import necessary libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.models.neural_networks import MultiLayerPerceptron
from src.models.markov_models import HiddenMarkovModel
from src.models.hidden_markov_models import GaussianMixtureHMM
from src.data.dataset import EEGDataset
from src.data.preprocess import load_data, filter_data, segment_data, extract_features
from src.data.model_data import preprocess_data, create_sequences

def train_nn_model(data_path: str, sampling_rate: int, low_freq: float, high_freq: float, segment_length: int, segment_overlap: int, feature_type: str, hidden_units: Tuple[int, int], learning_rate: float, epochs: int, batch_size: int, test_size: float) -> Dict:
    """
    Trains a neural network on the preprocessed EEG signal data using the specified parameters.
    Returns a dictionary containing the trained model and its training and testing accuracy.
    """
    # load and preprocess data
    preprocessed_data = preprocess_data(data_path, sampling_rate, low_freq, high_freq, segment_length, segment_overlap, feature_type)
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    
    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # create neural network model
    input_shape = X_train.shape[1]
    output_shape = len(np.unique(y_train))
    nn_model = MultiLayerPerceptron(input_shape, hidden_units, output_shape)
    
    # train neural network model
    nn_model.train(X_train, y_train, learning_rate, epochs, batch_size)
    
    # evaluate neural network model on training and testing data
    train_preds = nn_model.predict(X_train)
    test_preds = nn_model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    # create and return dictionary containing model and accuracy scores
    model_dict = {
        'model_type': 'neural_network',
        'model': nn_model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    return model_dict


def train_hmm_model(data_path: str, sampling_rate: int, low_freq: float, high_freq: float, segment_length: int, segment_overlap: int, feature_type: str, num_states: int, num_mixtures: int, test_size: float) -> Dict:
    """
    Trains a Hidden Markov Model on the preprocessed EEG signal data using the specified parameters.
    Returns a dictionary containing the trained model and its testing accuracy.
    """
    # load and preprocess data
    preprocessed_data = preprocess_data(data_path, sampling_rate, low_freq, high_freq, segment_length, segment_overlap, feature_type)
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    
    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    create sequences for HMM training and testing
train_sequences, train_lengths = create_sequences(X_train, y_train)
test_sequences, test_lengths = create_sequences(X_test, y_test)

train HMM model on training data
hmm = HiddenMarkovModel(n_states=5, n_features=20, covariance_type='diag')
hmm.fit(train_sequences, train_lengths)

evaluate HMM model on testing data
test_accuracy = hmm.score(test_sequences, test_lengths)

print(f"HMM classification accuracy: {test_accuracy}")


