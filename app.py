import streamlit as st
import torch
import torchaudio
import os
import io
import numpy as np
from st_audiorec import st_audiorec # Import the stable recorder

# --- Step 1: Define Model Architecture & Load Expert Model ---
@st.cache_resource
def load_model():
    # A. Define the Model Architecture
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.AvgPool2d(kernel_size=2)
        )

    class AudioCNN(torch.nn.Module):
        def __init__(self, num_classes=7):
            super(AudioCNN, self).__init__()
            self.conv1 = conv_block(2, 32, kernel_size=(3,5), stride=(2,2), padding=(2,2))
            self.conv2 = conv_block(32, 64, kernel_size=(3,5), stride=(1,1), padding=(2,2))
            self.conv3 = conv_block(64, 128, kernel_size=(5,5), stride=(1,1), padding=(2,2))
            self.conv4 = conv_block(128, 256, kernel_size=(5,5), stride=(1,1), padding=(2,2))
            self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
            x = self.adaptive_pool(x); x = self.flatten(x); x = self.fc1(x)
            return x

    # B. Load the trained model weights
    model_path = "audio_classifier_model.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found. The app may be building. Please refresh in a moment.")
        st.stop()
        
    model = AudioCNN(num_classes=7)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Step 2: Define the Prediction Function ---
AV_CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'engine_idling', 'gun_shot', 'siren']
inv_class_map = {i: name for i, name in enumerate(AV_CLASSES)}

def predict(waveform, sr):
    # Preprocess the waveform
    if waveform.shape[0] == 1: waveform = torch.cat([waveform, waveform], dim=0)
    target_sr = 44100; num_samples = target_sr * 4
    if sr != target_sr: waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[1] > num_samples: waveform = waveform[:, :num_samples]
    else: waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.shape[1]))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=64)
    spectrogram = mel_transform(waveform)
    mean, std = spectrogram.mean(), spectrogram.std()
    spectrogram = (spectrogram - mean) / (std + 1e-6)
    spectrogram = spectrogram.unsqueeze(0).to(torch.device("cpu"))
    
    # Make prediction
    with torch.no_grad():
        output = model(spectrogram)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidences = {AV_CLASSES[i]: float(probabilities[i]) for i in range(len(AV_CLASSES))}
        
    return confidences

# --- Step 3: Build the Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("Urban Sound Classifier 🔊")
st.write("This app classifies urban sounds using a CNN model trained on the UrbanSound8K dataset.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        waveform, sr = torchaudio.load(io.BytesIO(bytes_data))
        st.audio(bytes_data, format='audio/wav')

        if st.button("Classify Uploaded Audio"):
            with st.spinner('Analyzing sound...'):
                confidences = predict(waveform, sr)
                # Find the top prediction
                prediction = max(confidences, key=confidences.get)
                st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                st.write("--- Confidence Scores ---")
                for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")

with col2:
    st.header("Record Audio from Microphone")
    wav_audio_data = st_audiorec() # This creates the recorder widget

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        
        if st.button("Classify Recorded Audio"):
            waveform, sr = torchaudio.load(io.BytesIO(wav_audio_data))
            with st.spinner('Analyzing sound...'):
                confidences = predict(waveform, sr)
                # Find the top prediction
                prediction = max(confidences, key=confidences.get)
                st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                st.write("--- Confidence Scores ---")
                for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")

