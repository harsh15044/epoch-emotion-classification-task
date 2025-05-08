
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# Simple argument checking
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} audio_file.wav")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
audio_file = sys.argv[1]


# Load emotion mappings
mappings=  os.path.join(script_dir, '..', 'training', 'emotion_mappings.json')
with open(mappings, 'r') as f:
    mappings = json.load(f)
    id_to_emotion = {int(k): v for k, v in mappings['id_to_emotion'].items()}

# Audio preprocessing functions
def load_audio(file_path, sr=22050, duration=3):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    if len(y) < duration * sr:
        y = np.pad(y, (0, duration * sr - len(y)), 'constant')
    y = librosa.util.normalize(y)
    return y, sr

def extract_mfcc(y, sr, n_mfcc=40, n_fft=2048, hop_length=512, max_length=130):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    return mfcc

def extract_melspec(y, sr, n_fft=2048, hop_length=512, n_mels=128, max_length=130):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = (melspec - np.mean(melspec)) / np.std(melspec)
    if melspec.shape[1] < max_length:
        melspec = np.pad(melspec, ((0, 0), (0, max_length - melspec.shape[1])), mode='constant')
    else:
        melspec = melspec[:, :max_length]
    return melspec

# CNN model architecture
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 5 * 16, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     "cpu")

print(f"Using device: {device}")

# Load the model
model = AudioCNN().to(device)
model_path = os.path.join(script_dir, '..', 'training', 'audio_emotion_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load and process audio
y, sr = load_audio(audio_file)
mfcc = extract_mfcc(y, sr)
melspec = extract_melspec(y, sr)

# Make sure melspec has the right shape
if melspec.shape[0] != 40:
    melspec = librosa.util.fix_length(melspec, size=40, axis=0)

# Stack features and convert to tensor
features = np.stack([mfcc, melspec], axis=0)
features = torch.FloatTensor(features).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    outputs = model(features)
    probs = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)

# Get results
emotion_id = predicted.item()
emotion = id_to_emotion[emotion_id]
confidence = probs[0][emotion_id].item()

# Print results
print(f"\nPredicted emotion: {emotion} with {confidence*100:.2f}% confidence")

# Print all probabilities
print("\nAll emotion probabilities:")
for i, prob in enumerate(probs[0].cpu().numpy()):
    if i in id_to_emotion:
        print(f"{id_to_emotion[i]}: {prob*100:.2f}%")