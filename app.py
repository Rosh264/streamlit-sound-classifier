import streamlit as st
import torch
import torchaudio
import os
import numpy as np
from scipy.io.wavfile import read as read_wav
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import io

# --- Step 1: Define the Model Architecture and Load the Expert Model ---
# This part is loaded only once, thanks to Streamlit's caching.

@st.cache_resource
def load_model():
    # A. Define the Model Architecture (copy-pasted from your training code)
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
        st.error("The model file 'audio_classifier_model.pth' was not found.")
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
        # Find the class with the highest probability
        prediction_index = output.argmax(dim=1).item()
        predicted_class = inv_class_map[prediction_index]
        
    return predicted_class, confidences

# --- Step 3: Build the Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("Urban Sound Classifier 🔊")
st.write("A web app to classify urban sounds, replicating the paper 'Improving the Environmental Perception of Autonomous Vehicles'.")
st.write("Upload an audio file or use the live recorder to classify a sound into one of 7 categories.")

# Create columns for the layout
col1, col2 = st.columns(2)

with col1:
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To convert to a tensor
        waveform, sr = torchaudio.load(io.BytesIO(bytes_data))
        
        st.audio(bytes_data, format='audio/wav')

        if st.button("Classify Uploaded File"):
            with st.spinner('Analyzing sound...'):
                prediction, confidences = predict(waveform, sr)
                st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                st.write("--- Confidence Scores ---")
                for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")

with col2:
    st.header("Live Audio Recorder")
    st.write("Click 'start' to record a 4-second clip. The prediction will appear automatically.")

    # This class handles processing audio from the microphone
    class AudioRecorder(AudioProcessorBase):
        def __init__(self) -> None:
            self.audio_buffer = []

        def recv(self, frame):
            # The audio comes in frames, we buffer them
            self.audio_buffer.append(frame.to_ndarray())
            return frame

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SEND_ONLY,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
    )

    if not webrtc_ctx.state.playing:
        st.write("Recorder is off.")
    else:
        st.write("Recording...")
        
    if st.button("Classify Recording"):
        if webrtc_ctx.audio_processor:
            audio_frames = webrtc_ctx.audio_processor.audio_buffer
            if audio_frames:
                # Combine the audio frames into a single numpy array
                sound_chunk = np.concatenate(audio_frames, axis=1)
                # The sample rate from the browser is usually 48000
                sr = 48000 
                
                # Convert to a torch tensor
                waveform = torch.from_numpy(sound_chunk).float()

                with st.spinner('Analyzing sound...'):
                    prediction, confidences = predict(waveform, sr)
                    st.success(f"**Prediction:** {prediction.replace('_', ' ').title()}")
                    st.write("--- Confidence Scores ---")
                    for class_name, prob in sorted(confidences.items(), key=lambda item: item[1], reverse=True):
                        st.write(f"{class_name.replace('_', ' ').title()}: {prob:.2%}")
                
                # Clear the buffer
                webrtc_ctx.audio_processor.audio_buffer = []
            else:
                st.warning("No audio recorded yet. Please start the recorder and make a sound.")
        else:
            st.error("Audio recorder is not ready.")