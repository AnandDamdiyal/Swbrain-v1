import numpy as np
import pandas as pd
import os


def save_data(data: np.ndarray, file_path: str) -> None:
    """
    Saves data to the specified file path.
    """
    np.save(file_path, data)


def load_data(file_path: str) -> np.ndarray:
    """
    Loads data from the specified file path.
    """
    return np.load(file_path)


def save_dataframe(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Saves pandas DataFrame to the specified file path.
    """
    dataframe.to_csv(file_path, index=False)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads pandas DataFrame from the specified file path.
    """
    return pd.read_csv(file_path)


def create_directory(directory_path: str) -> None:
    """
    Creates a directory at the specified path.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def delete_directory(directory_path: str) -> None:
    """
    Deletes the directory at the specified path.
    """
    if os.path.exists(directory_path):
        os.rmdir(directory_path)


def split_data(data: np.ndarray, labels: np.ndarray, split_ratio: float) -> tuple:
    """
    Splits data and corresponding labels into training and testing sets
    according to the specified split ratio.
    """
    split_index = int(len(data) * split_ratio)
    train_data, test_data = data[:split_index], data[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    return train_data, train_labels, test_data, test_labels


def balance_data(data: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Balances data by undersampling the majority class and oversampling the minority class.
    Returns the balanced data and corresponding labels.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]
    minority_class = unique_labels[np.argmin(counts)]
    majority_indices = np.where(labels == majority_class)[0]
    minority_indices = np.where(labels == minority_class)[0]
    num_samples = min(counts)
    balanced_indices = np.concatenate([majority_indices[:num_samples], minority_indices[:num_samples]])
    np.random.shuffle(balanced_indices)
    balanced_data = data[balanced_indices]
    balanced_labels = labels[balanced_indices]
    return balanced_data, balanced_labels
