#imports
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display 
import os
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Download latest version of dataset
path_dataset = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

#use GPU
if torch.cuda.is_available():
    device= torch.device("cuda")
    print("Using Cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    print("Using mps")
else:
    device = torch.device("cpu")
    print("Using CPU")

#functions
def get_file_meta(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('-')

    emotion_mappings = {
        '01': 'neutral',
        '02': 'calm', 
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    emotion_id = parts[2]
    emotion = emotion_mappings[emotion_id]
    actor = parts[6].split('.')[0]
    gender = 'male' if int(actor)%2 else 'female'

    return {
        'filepath' : filepath,
        'emotion_id': emotion_id,
        'emotion' : emotion, 
        'actor': actor,
        'gender':gender
    }


#get fikes
ravdess_files = []

#we are skipping the folder named 'audio_speech_actors_01-24' as it also has the same files.
for actor_id in range(1, 25):  #Actor_01 to Actor_24
    actor_folder = os.path.join(path_dataset, f"Actor_{actor_id:02d}")
    for file in os.listdir(actor_folder):
        if file.endswith('.wav'):
            ravdess_files.append(os.path.join(actor_folder, file))

print(f"Total audio files: {len(ravdess_files)}")

#convert to df
df = pd.DataFrame([get_file_meta(file) for file in ravdess_files])


#functions for preprocessing
#to load audio in a format which we can work on
def load_audio(file_path, sr=22050, duration=3):
    y, sr = librosa.load(file_path, sr=sr,duration=duration)
    if len(y)<duration*sr:
        y = np.pad(y, (0, duration*sr - len(y)), 'constant')
    y = librosa.util.normalize(y)
    return y, sr
    

#Mel-frequency Cepstral Coefficient extraction
def extract_mfcc(y,sr, n_mfcc=40, n_fft=2048, hop_length=512,max_length=130):
    mfcc = librosa.feature.mfcc(y=y,sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length = hop_length)
    
    #normalizing
    mfcc = (mfcc- np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0,0), (0,max_length - mfcc.shape[1])), mode ='constant')
    else:
        mfcc=mfcc[:, :max_length]
    return mfcc

#mel-spectrogram extraction (considered good for emotion detection (source: online articles))
def extract_melspec(y,sr,n_fft=2048,hop_length=512, n_mels=128, max_length=130 ):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    #normalization
    melspec = (melspec-np.mean(melspec)) / np.std(melspec)

    if melspec.shape[1] < max_length:
        melspec = np.pad(melspec, ((0,0), (0,max_length - melspec.shape[1])), mode='constant')
    else:
        melspec = melspec[:, :max_length]
    return melspec


#augmentation
def augment_audio(y,sr):

    n_aug = np.random.randint(0,3)

    for _ in range(n_aug):
        aug = np.random.choice(['time_shift', 'noise', 'pitch_shift', 'speed'])

        if aug == 'time_shift':
            shift = int(np.random.uniform(-0.1,0.1)* len(y))
            y = np.roll(y,shift)

        elif aug=='pitch_shift':
            n_steps = np.random.uniform(-3,3)
            y = librosa.effects.pitch_shift(y,sr=sr, n_steps=n_steps)
        
        elif aug=='noise':
            noise_level= np.random.uniform(0.001, 0.01)
            y = y + noise_level*np.random.randn(len(y))

        elif aug=="speed":
            factor = np.random.uniform(0.8,1.2)
            y=librosa.effects.time_stretch(y, rate=factor)

            if(len(y)>sr*3):
                y=y[:sr*3]
            else:
                y = np.pad(y,(0,max(0,sr*3 - len(y))), mode='constant')

    y = librosa.util.normalize(y)
    return y


#dataset
class RavdessDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['filepath']
        emotion_id = self.df.iloc[idx]['emotion']
        label= emotion_to_id[emotion_id]
        y,sr = load_audio(path)

        if self.augment:
            y = augment_audio(y,sr)

        
        mfcc = extract_mfcc(y,sr)
        melspec= extract_melspec(y,sr)

        #making size same
        if melspec.shape[0]!=40:
            melspec= librosa.util.fix_length(melspec,size=40,axis=0)
        
        features = np.stack([mfcc,melspec], axis=0)
        features = torch.FloatTensor(features)
        return features,label


#dataloaders

#Splitting the dataset
train_df, val_df= train_test_split(df, test_size=0.3, stratify=df['emotion'])
emotion_to_id= {emotion: idx for idx, emotion in enumerate(df['emotion'].unique())}
id_to_emotion= {idx: emotion for emotion, idx in emotion_to_id.items()}

train_dataset= RavdessDataset(train_df, augment=True)
val_dataset= RavdessDataset(val_df, augment=False)

batch_size =32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#architecture
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        #layer1 (2 input channels melspec and mfcc)
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


    def forward(self,x):
        batch_size = x.size(0)

        #TODO: specaugment
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x



#training loop function
def train_model(model, train_loader, val_loader,criterion, optimizer, scheduler, num_epochs=30):
    train_losses= []
    val_losses= []
    train_accs= []
    val_accs= []

    best_val_loss = float('inf')
    best_train_loss= 0
    best_val_acc=0
    best_train_acc=0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)


        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate epoch metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        
        # Update learning rate
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_train_loss = epoch_train_loss
            best_train_acc = epoch_train_acc
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()

        print(f"Completed Epoch {epoch+1}/{num_epochs}")
        if(epoch==num_epochs-1):
            print(f"Best Model information")
            print(f"Train Loss:{best_train_loss:.4f}, Train Accuracy: {best_train_acc:.4f}%")
            print(f"Val Loss:  {best_val_loss:.4f},   Val Accuracy: {best_val_acc:.4f}%")


    model.load_state_dict(best_model_state)
    print("Training complete! Loaded the best model.")
    
    return model, train_losses, val_losses, train_accs, val_accs


#training
model = AudioCNN().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr = 0.0003, weight_decay=0.002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5,  min_lr=0.00001)

# Train the model
model, train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30
)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'audio_emotion_model.pth')
mappings_path = os.path.join(script_dir, 'emotion_mappings.json')

#saving the model
torch.save(model.state_dict(), model_path)

# Save emotion mappings
with open(mappings_path, 'w') as f:
    json.dump({
        'emotion_to_id': {emotion: idx for emotion, idx in emotion_to_id.items()},
        'id_to_emotion': {str(idx): emotion for idx, emotion in id_to_emotion.items()}
    }, f)

print("Model and mappings saved successfully")