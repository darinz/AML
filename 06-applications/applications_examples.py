"""
Applications & Deployment Examples

This file contains examples for:
- Audio and 1D signal processing
- Model serving with Flask
- Time series analysis
- Reinforcement learning (Q-learning)
- Model monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
from flask import Flask, request, jsonify
import pickle
import gym

# 1. Audio and Signal Processing Example
fs = 1000
T = 1.0
t = np.linspace(0, T, int(fs*T), endpoint=False)
sig = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
fft = np.fft.fft(sig)
freqs = np.fft.fftfreq(len(sig), 1/fs)
plt.plot(freqs[:len(sig)//2], np.abs(fft)[:len(sig)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Signal')
plt.show()

# 2. Low-pass Filter Example
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
filtered = lowpass_filter(sig, 100, fs)
plt.plot(t, sig, label='Original')
plt.plot(t, filtered, label='Filtered')
plt.legend()
plt.title('Low-pass Filter')
plt.show()

# 3. Spectrogram Example
f, tt, Sxx = spectrogram(sig, fs)
plt.pcolormesh(tt, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='dB')
plt.show()

# 4. Model Serving Example (Flask)
# (This is a minimal example; run with Flask for real serving)
# Save a dummy model
import sklearn.linear_model
model = sklearn.linear_model.LogisticRegression().fit(np.array([[0, 1], [1, 0]]), [0, 1])
pickle.dump(model, open('model.pkl', 'wb'))
# Flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})
# To run: app.run()

# 5. Q-Learning Example (OpenAI Gym)
try:
    import gym
    env = gym.make('FrozenLake-v1', is_slippery=False)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for ep in range(100):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(Q[s]) if np.random.rand() > 0.1 else env.action_space.sample()
            s_, r, done, _, = env.step(a)
            Q[s, a] += 0.1 * (r + 0.99 * np.max(Q[s_]) - Q[s, a])
            s = s_
    print("Q-table after training:", Q)
except Exception as e:
    print("Install gym for RL example.")

# 6. Monitoring Example (Dummy)
metrics = {'accuracy': 0.95, 'latency_ms': 50}
print("Model metrics:", metrics) 