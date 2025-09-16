import streamlit as st
import torch
import torchaudio
import os
import io
import numpy as np
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt

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

# --- Step 2: Define the Prediction and Visualization Functions ---
AV_CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'engine_idling', 'gun_shot', 'siren']
inv_class_map = {i: name for i, name in enumerate(AV_CLASSES)}

def preprocess_audio(waveform, sr):
    if waveform.shape[0] == 1: 
        waveform = torch.cat([waveform, waveform], dim=0)
    target_sr = 44100
    num_samples = target_sr * 4
    if sr != target_sr: 
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[1] > num_samples: 
        waveform = waveform[:, :num_samples]
    else: 
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.shape[1]))
    return waveform, target_sr

def create_spectrogram(waveform, sr):
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=64)
    return mel_transform(waveform)

def predict(spectrogram):
    mean, std = spectrogram.mean(), spectrogram.std()
    spectrogram = (spectrogram - mean) / (std + 1e-6)
    spectrogram = spectrogram.unsqueeze(0).to(torch.device("cpu"))
    
    with torch.no_grad():
        output = model(spectrogram)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidences = {AV_CLASSES[i]: float(probabilities[i]) for i in range(len(AV_CLASSES))}
        
    return confidences

def plot_waveform(waveform, sr, title="Waveform"):
    waveform_numpy = waveform.numpy()
    num_channels, num_frames = waveform_numpy.shape
    time_axis = np.linspace(0, num_frames / sr, num=num_frames)

    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3))
    plt.style.use('dark_background')
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        axes[i].plot(time_axis, waveform_numpy[i], linewidth=1)
        axes[i].grid(True)
        if num_channels > 1:
            axes[i].set_ylabel(f"Channel {i+1}")
    fig.suptitle(title)
    return fig

def plot_spectrogram(specgram, title="Mel Spectrogram"):
    fig = plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    plt.imshow(torchaudio.transforms.AmplitudeToDB()(specgram)[0].numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    return fig

# --- Step 3: Build the Streamlit User Interface ---

st.set_page_config(layout="wide", page_title="Urban Sound Classifier")
st.title("Urban Sound Classifier 🔊")
st.write("This app classifies urban sounds using a CNN model trained on the UrbanSound8K dataset.")
st.write("---")

# Use session state to store the recorded audio
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

col1, col2 = st.columns(2)

with col1:
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Classify Uploaded Audio"):
            with st.spinner('Analyzing sound...'):
                bytes_data = uploaded_file.getvalue()
                waveform, sr = torchaudio.load(io.BytesIO(bytes_data))
                processed_waveform, processed_sr = preprocess_audio(waveform, sr)
                spectrogram = create_spectrogram(processed_waveform, processed_sr)
                
                st.pyplot(plot_waveform(processed_waveform, processed_sr))
                st.pyplot(plot_spectrogram(spectrogram))

                confidences = predict(spectrogram)
                prediction = max(confidences, key=confidences.get)
                st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                st.write("--- Confidence Scores ---")
                for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")

with col2:
    st.header("Record Audio from Microphone")
    st.write("1. Click Start Recording. 2. Make a sound. 3. Click Stop. 4. Your recording will appear below with a classify button.")
    
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # Store the recorded audio in session state
        st.session_state.recorded_audio = wav_audio_data
    
    # Display the audio player and classify button only after a recording is made
    if st.session_state.recorded_audio is not None:
        st.audio(st.session_state.recorded_audio, format='audio/wav')
        
        if st.button("Classify Recorded Audio"):
            waveform, sr = torchaudio.load(io.BytesIO(st.session_state.recorded_audio))
            with st.spinner('Analyzing sound...'):
                processed_waveform, processed_sr = preprocess_audio(waveform, sr)
                spectrogram = create_spectrogram(processed_waveform, processed_sr)
                
                st.pyplot(plot_waveform(processed_waveform, processed_sr))
                st.pyplot(plot_spectrogram(spectrogram))

                confidences = predict(spectrogram)
                prediction = max(confidences, key=confidences.get)
                st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                st.write("--- Confidence Scores ---")
                for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")

    st.info(
        "Note: Microphone recording can be unreliable on some desktop browsers due to strict security policies. "
        "If you encounter issues, please try a different browser or use the file upload option."
    )

