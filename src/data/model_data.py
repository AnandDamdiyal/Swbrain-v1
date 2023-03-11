import numpy as np
from typing import Tuple, List

class ModelData:
    """
    Class to preprocess and prepare data for machine learning models.
    """
    def __init__(self, sampling_rate: int, low_freq: float, high_freq: float, segment_length: int, segment_overlap: int):
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data from the given file path and returns the raw EEG signal data
        and corresponding labels.
        """
        # code to load data from file and return as tuple of data and labels
        return data, labels

    def filter_data(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Applies a bandpass filter to the raw EEG signal data in order to remove noise
        and unwanted frequencies.
        """
        # code to apply a bandpass filter to the raw data
        return filtered_data

    def segment_data(self, filtered_data: np.ndarray) -> List[np.ndarray]:
        """
        Segments the filtered EEG signal data into smaller, non-overlapping segments of
        equal length for use in further analysis.
        """
        # code to segment the filtered data into smaller segments
        return segmented_data

    def extract_features(self, segmented_data: List[np.ndarray], feature_type: str) -> np.ndarray:
        """
        Extracts features from the segmented EEG signal data using the specified feature
        extraction method.
        """
        # code to extract features from the segmented data using the specified method
        return feature_data

    def prepare_data(self, data_path: str, feature_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses and prepares the data for machine learning models by calling the necessary
        functions in the correct order. Returns a tuple of preprocessed feature data and labels.
        """
        # code to call the necessary preprocessing functions in the correct order
        return feature_data, labels
