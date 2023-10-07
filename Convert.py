# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64):
#     # Load the image and convert to grayscale
#     img = Image.fromarray(image).convert("L")
    
#     # Calculate spectrogram size
#     spec_shape = (int(librosa.time_to_frames(time, sr=rate, hop_length=n_fft//2, n_fft=n_fft)), n_fft)
    
#     # Resize the image to match the spectrogram shape
#     img = np.asarray(img.resize(spec_shape))
    
#     # Normalize the image
#     img = (img - img.min()) / (img.max() - img.min())
    
#     # Convert to magnitude spectrogram
#     spec = librosa.db_to_amplitude(librosa.feature.melspectrogram(S=img, sr=rate, n_fft=n_fft))
    
#     # Convert to PyTorch tensor
#     spec_tensor = torch.tensor(spec)
    
#     # Convert the spectrogram to audio using Griffin-Lim
#     mel_scale = torchaudio.transforms.MelScale(n_stft=n_fft, n_mels=spec.shape[0], sample_rate=rate)
#     mel_inverse = torchaudio.transforms.InverseMelScale(n_stft=n_fft, n_mels=spec.shape[0], sample_rate=rate)
    
#     audio = mel_inverse(mel_scale(spec_tensor))
    
#     return rate, audio.numpy()[0]

# time = gr.Number(3.0, label="Audio time")
# image = gr.Image(label="Image to sonify")
# n_fft = gr.Number(1024, label="n_fft")

# def main(image, time, n_fft):
#     return img_to_audio(image, time=time, rate=22050, n_fft=int(n_fft))

# desc = "Upload an image you would like to hear."

# interface = gr.Interface(fn=main, inputs=[image, time, n_fft], outputs="audio", title="Improved Image Sonification", description=desc)

# interface.launch()

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
from PIL import Image

def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64):
    # Load the image and convert to grayscale
    img = Image.fromarray(image).convert("L")
    
    # Calculate spectrogram size
    spec_shape = (int(librosa.time_to_frames(time, sr=rate, hop_length=n_fft//2, n_fft=n_fft)), n_fft)
    
    # Resize the image to match the spectrogram shape
    img = np.asarray(img.resize(spec_shape))
    
    # Normalize the image
    img = (img - img.min()) / (img.max() - img.min())
    
    # Convert to magnitude spectrogram
    spec = librosa.db_to_amplitude(librosa.feature.melspectrogram(S=img, sr=rate, n_fft=n_fft))
    
    # Convert to PyTorch tensor
    spec_tensor = torch.tensor(spec)
    
    # Convert the spectrogram to audio using Griffin-Lim
    mel_scale = torchaudio.transforms.MelScale(n_stft=n_fft, n_mels=spec.shape[0], sample_rate=rate)
    mel_inverse = torchaudio.transforms.InverseMelScale(n_mels=spec.shape[0], sample_rate=rate, n_stft=n_fft)
    
    spec_mel = mel_scale(spec_tensor)
    
    if spec_mel.shape[2] > 0:
        spec_mel_t = np.transpose(spec_mel, (0, 2, 1))
        spec_mel_t = spec_mel_t.reshape(spec_mel_t.shape[0] * spec_mel_t.shape[1], spec_mel_t.shape[2])
        
        audio = mel_inverse(spec_mel_t.unsqueeze(0)).squeeze(0)
    else:
        audio = np.zeros((1,))
    
    return rate, audio.numpy()

time = gr.Number(3.0, label="Audio time")
image = gr.Image(label="Image to sonify")
n_fft = gr.Number(1024, label="n_fft")

def main(image, time, n_fft):
    return img_to_audio(image, time=time, rate=22050, n_fft=int(n_fft))

desc = "Upload an image you would like to hear."

interface = gr.Interface(fn=main, inputs=[image, time, n_fft], outputs="audio", title="Improved Image Sonification", description=desc)

interface.launch()
