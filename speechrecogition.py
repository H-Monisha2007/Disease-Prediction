import os
import torch
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim

# ======================================
# 1️⃣ MFCC Feature Extraction
# ======================================

def pre_emphasis(signal, coeff=0.97):
    """Apply pre-emphasis filter."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def mfcc(signal, sample_rate, num_ceps=40, nfft=512, hop_length=256):
    """
    Compute MFCC features from audio signal.
    """
    # Pre-emphasis
    emphasized = pre_emphasis(signal)

    # Framing
    frame_length = 0.025 * sample_rate  # 25ms
    frame_step = 0.010 * sample_rate    # 10ms
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window
    frames *= np.hamming(frame_length)

    # FFT and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))

    # Mel filter banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1] + 1e-6)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m] + 1e-6)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # DCT to get MFCCs
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Take mean across time frames
    mfccs_mean = np.mean(mfccs, axis=0)

    return mfccs_mean

# ======================================
# 2️⃣ Load Dataset
# ======================================

data_path = r"C:\Users\monis\AppData\Local\Programs\Python\Python39\AudioData\AudioData"  # Change your path
features = []
labels = []

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # Read audio
            try:
                sample_rate, signal = wavfile.read(file_path)
            except:
                print(f"Skipping unreadable file: {file_path}")
                continue

            # Convert to mono if stereo
            if len(signal.shape) > 1:
                signal = signal.mean(axis=1)

            # Extract MFCC
            mfcc_features = mfcc(signal, sample_rate, num_ceps=40)
            features.append(mfcc_features)

            # Extract emotion
            parts = file.split("-")
            if len(parts) >= 3:
                emotion = parts[2]  # RAVDESS format
            else:
                emotion = os.path.basename(root)
            labels.append(emotion)

X = np.array(features)
y = np.array(labels)

print(f"Total samples: {len(X)}, Features shape: {X.shape}, Labels shape: {y.shape}")

# ======================================
# 3️⃣ Encode Labels
# ======================================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ======================================
# 4️⃣ Define Neural Network
# ======================================

class EmotionNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = EmotionNet(input_size=40, num_classes=len(le.classes_))

# Weighted loss to handle imbalance
class_counts = np.bincount(y_encoded)
class_weights = 1. / (class_counts + 1e-6)
weights = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================================
# 5️⃣ Train Model
# ======================================

epochs = 50
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ======================================
# 6️⃣ Evaluate
# ======================================

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

print("\nClassification Report:\n")
print(classification_report(y_test.numpy(), predicted.numpy(), target_names=le.classes_))