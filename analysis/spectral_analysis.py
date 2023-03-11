import numpy as np
import scipy.signal as signal

def compute_power_spectrum(signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Computes the power spectrum of a given signal using Welch's method.
    Returns an array of the power spectral density values.
    """
    # define window size and overlap for Welch's method
    window_size = int(sampling_rate * 2)
    overlap = int(window_size / 2)
    
    # compute power spectral density using Welch's method
    _, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
    
    return psd

def compute_spectrogram(signal_data: np.ndarray, sampling_rate: int, window_size: float, overlap: float, nfft: int) -> np.ndarray:
    """
    Computes the spectrogram of a given signal using a sliding window approach.
    Returns a 2D array of spectrogram values.
    """
    # compute spectrogram using sliding window approach
    _, _, spectrogram = signal.spectrogram(signal_data, fs=sampling_rate, window='hann', nperseg=int(window_size*sampling_rate), noverlap=int(overlap*sampling_rate), nfft=nfft)
    
    return spectrogram

def compute_coherence(signal_data_1: np.ndarray, signal_data_2: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Computes the coherence between two given signals using Welch's method.
    Returns an array of the coherence values.
    """
    # define window size and overlap for Welch's method
    window_size = int(sampling_rate * 2)
    overlap = int(window_size / 2)
    
    # compute coherence using Welch's method
    _, coherence = signal.coherence(signal_data_1, signal_data_2, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
    
    return coherence
