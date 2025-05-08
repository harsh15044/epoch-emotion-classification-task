# Audio Emotion Recognition
This project uses deep learning to recognize emotions from audio files.



## Overview

The model has been trained on the [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), which contains recordings of various actors expressing various emotions. It uses a combination of mel-frequency cepstral coefficients (MFCC) and mel-spectrograms to extract features from audio, which are then fed into a convolutional neural network for classification.

## Features

- Emotion recognition from WAV audio files
- Support for 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised
- Detailed probability output for each emotion
- GPU acceleration utilized when available

## Installation

### Using Poetry (Recommended)

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. First, install Poetry if you don't have it already:
```bash
pip3 install poetry
```

2. Clone the repository:
```bash
git clone https://github.com/harsh15044/audio-emotion-recognition.git

cd audio-emotion-recognition
```

3. Install dependencies with Poetry:
```bash
poetry install
```

4. Train the Model:
To explore the training process interactively, you may use the provided Jupyter notebook. Alternatively, to quickly train the model, run the Python script below:
```bash
poetry run python training/train.py
```


5. To run the inference script:
```bash
poetry run python inference/infer.py path/to/audio_file.wav
```

Sample audio files are available in `inference/samples`.

### Manual Installation

If you prefer not to use Poetry:

```bash
pip3 install -r requirements.txt
```

## Project Structure

```
CNN/
├── training/                      # Model training code and saved model
│   ├── audio_emotion_model.pth    # Trained PyTorch model weights
│   └── emotion_mappings.json      # Mapping between emotion IDs and labels
│   └── train.py                   # Script to train and save the model
│   └── train_model.ipynb          # Jupyter notebook to explore the training process
├── inference/                     # Inference code
│   └── infer.py                   # Script for making predictions
│   └── samples/                   # Sample files for inferencing
├── pyproject.toml                 # Poetry configuration and dependencies
└── README.md                  
```

## Usage

To predict the emotion in an audio file:

```bash
# If using Poetry
poetry run python inference/infer.py path/to/audio_file.wav

# If not using Poetry
python inference/infer.py path/to/audio_file.wav
```


Example usage:
```
poetry run python inference/infer.py inference/samples/happy.wav
```

Example output:
```
Using device: mps

Predicted emotion: happy with 87.25% confidence

All emotion probabilities:
neutral: 2.15%
calm: 1.43%
happy: 87.25%
sad: 0.89%
angry: 3.67%
fearful: 2.31%
disgust: 0.78%
surprised: 1.52%
```


## Model Architecture

The model uses a CNN architecture with:
- Two input channels (MFCC and mel-spectrogram)
- Three convolutional blocks with batch normalization and max pooling
- Fully connected layers with dropout for regularization
- Output layer with 8 units (one for each emotion)

The model is implemented in PyTorch and relies on librosa for audio preprocessing.

## Training

The model was trained on the RAVDESS dataset with data augmentation techniques including:
- Time shifting
- Pitch shifting
- Adding noise
- Speed variation

## Performance

The model achieves approximately 70% accuracy on the validation set. Performance varies by emotion, with some emotions being recognized more accurately than others.

## Limitations

- Short audio clips (less than 3 seconds) might not contain enough information for accurate classification
- The model was trained primarily on acted emotions, so performance may vary with natural emotional expressions

## References

- RAVDESS Dataset: available [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

