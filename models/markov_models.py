import numpy as np

class MarkovModel:
    """
    Class for implementing a Markov model for time series data.
    """

    def __init__(self, n_states=2):
        self.n_states = n_states
        self.transition_matrix = None

    def fit(self, X, y=None):
        """
        Fits the model to the training data by estimating the transition matrix.
        """
        # Convert time series data to state sequences
        sequences = self._to_sequences(X)

        # Estimate transition matrix from state sequences
        transition_counts = np.zeros((self.n_states, self.n_states))
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                transition_counts[current_state, next_state] += 1
        self.transition_matrix = self._normalize_rows(transition_counts)

    def predict(self, X):
        """
        Predicts the state sequence for each time series in X using the fitted model.
        """
        # Convert time series data to state sequences
        sequences = self._to_sequences(X)

        # Predict state sequence for each time series
        y_pred = []
        for sequence in sequences:
            current_state = sequence[0]
            state_sequence = [current_state]
            for i in range(len(sequence) - 1):
                next_state_probs = self.transition_matrix[current_state, :]
                next_state = np.argmax(next_state_probs)
                state_sequence.append(next_state)
                current_state = next_state
            y_pred.append(np.array(state_sequence))

        return np.array(y_pred)

    def _to_sequences(self, X):
        """
        Converts time series data to sequences of discrete states using a simple
        threshold-based discretization method.
        """
        sequences = []
        for x in X:
            sequence = np.zeros(len(x))
            for i in range(len(x)):
                if x[i] >= np.mean(x):
                    sequence[i] = 1
            sequences.append(sequence.astype(int))
        return sequences

    def _normalize_rows(self, matrix):
        """
        Normalizes the rows of a matrix to sum to 1.
        """
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums[:, np.newaxis]
