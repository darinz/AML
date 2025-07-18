# Audio and 1D Signals

## Overview

This guide covers the fundamentals of processing audio and 1D signals for machine learning, including signal processing, feature extraction, classification, and time series analysis.

## 1. Signal Processing Basics

### Fourier Transform
- Converts a signal from time to frequency domain
- Discrete Fourier Transform (DFT):
```math
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N}
```
- Fast Fourier Transform (FFT): Efficient algorithm for DFT

#### Example: FFT in Python
```python
import numpy as np
import matplotlib.pyplot as plt
fs = 1000  # Sampling rate
T = 1.0    # Duration
t = np.linspace(0, T, int(fs*T), endpoint=False)
sig = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
fft = np.fft.fft(sig)
freqs = np.fft.fftfreq(len(sig), 1/fs)
plt.plot(freqs[:len(sig)//2], np.abs(fft)[:len(sig)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Signal')
plt.show()
```

### Filtering
- Remove noise or extract frequency bands
- Low-pass, high-pass, band-pass filters

#### Example: Low-pass filter
```python
from scipy.signal import butter, lfilter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
```

## 2. Audio Classification

### Feature Extraction
- **MFCCs**: Mel-frequency cepstral coefficients
- **Spectrograms**: Visual representation of frequency over time

#### Example: Spectrogram
```python
from scipy.signal import spectrogram
f, t, Sxx = spectrogram(sig, fs)
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='dB')
plt.show()
```

### Classification Pipeline
- Extract features (MFCC, spectrogram)
- Train classifier (e.g., SVM, Random Forest)

## 3. Time Series Analysis

### Forecasting
- Predict future values from past data
- ARIMA, LSTM, Prophet

### Anomaly Detection
- Detect unusual patterns in time series

## 4. Real-Time Processing
- Streaming audio analysis
- Applications: speech recognition, music classification

## Applications
- Speech recognition
- Music genre classification
- Anomaly detection in sensor data
- Financial time series forecasting

## Summary
- Signal processing transforms and filters data
- Feature extraction enables classification
- Time series analysis for forecasting and anomaly detection
- Real-time processing for streaming applications 