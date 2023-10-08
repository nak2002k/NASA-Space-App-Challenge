import streamlit as st
import numpy as np
import librosa
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import os
import soundfile as sf
from pydub import AudioSegment

def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
    # Load image
    img = Image.fromarray(image).convert("L")

    # Apply preprocessing techniques
    if contrast_stretch:
        img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
    if hist_equalize:
        img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

    # Calculate spectrogram size
    spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
    spec = np.asarray(img.resize(spec_shape))
    spec = np.interp(spec, (spec.min(), spec.max()), (-30, 10))  # Adjust the range
    spec = librosa.db_to_amplitude(spec)

    if improve_reconstruction:
        # Use advanced reconstruction method
        reconstructed_spec = librosa.feature.inverse.mel_to_audio(spec)
        audio = librosa.effects.preemphasis(reconstructed_spec)
    else:
        # Use Griffin-Lim for reconstruction with adjusted parameters
        audio = librosa.griffinlim(spec, n_iter=n_iter * 2, hop_length=hop_length)

    # Apply more smoothing to make the audio more calm
    audio = smooth_audio(audio, sigma=2)

    # Increase the amplitude to make the audio louder
    audio = 1.5 * audio  # Adjust the multiplier as needed

    return rate, audio

    # Load image
    img = Image.fromarray(image).convert("L")

    # Apply preprocessing techniques
    if contrast_stretch:
        img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
    if hist_equalize:
        img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

    # Calculate spectrogram size
    spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
    spec = np.asarray(img.resize(spec_shape))
    spec = np.interp(spec, (spec.min(), spec.max()), (-30, 10))  # Adjust the range
    spec = librosa.db_to_amplitude(spec)

    if improve_reconstruction:
        # Use advanced reconstruction method
        reconstructed_spec = librosa.feature.inverse.mel_to_audio(spec)
        audio = librosa.effects.preemphasis(reconstructed_spec)
    else:
        # Use Griffin-Lim for reconstruction with adjusted parameters
        audio = librosa.griffinlim(spec, n_iter=n_iter * 2, hop_length=hop_length)

    # Apply more smoothing to make the audio more calm
    audio = smooth_audio(audio, sigma=2)

    # Adjust the amplitude to make the audio calmer
    audio = 0.3 * audio

    return rate, audio

def smooth_audio(audio, sigma=1):
    # Apply Gaussian smoothing to the audio
    smoothed_audio = gaussian_filter(audio, sigma=sigma)
    return smoothed_audio

def generate_waveform(audio, rate):
    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio)) / rate, audio, color='b')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Display the waveform plot
    st.pyplot()

def read_video_frames(uploaded_file, frame_skip=1):
    # Save the uploaded video temporarily to a file
    with open("temp_video.mp4", "wb") as temp_video_file:
        temp_video_file.write(uploaded_file.read())

    # Open the temporarily saved video file
    cap = cv2.VideoCapture("temp_video.mp4")
    
    frames = []
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
           
            # Skip frames based on frame_skip value
            if frame_count % frame_skip == 0:
                frames.append(frame)
    except Exception as e:
        st.error(f"Error processing frames: {str(e)}")
    finally:
        # Close the video capture object and release the file
        cap.release()
    
    # Remove the temporary video file
    os.remove("temp_video.mp4")

    return frames

def video_to_audio(video_frames, output_audio_path, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
    audio_frames = []
    for frame in video_frames:
        audio = img_to_audio(frame, time, rate, n_fft, n_iter, hop_length, contrast_stretch, hist_equalize, improve_reconstruction)
        audio_frames.append(audio[1])
    audio_frames = np.concatenate(audio_frames)

    # Save the resulting audio as a WAV file
    sf.write(output_audio_path, audio_frames, rate)

def main():
    st.title("Improved Image and Video Sonification")

    time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
    n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
    hop_length = st.slider("hop_length", 256, 1024, 512, 64)
    n_iter = st.slider("n_iter", 10, 100, 64, 10)
    contrast_stretch = st.checkbox("Apply Contrast Stretching")
    hist_equalize = st.checkbox("Apply Histogram Equalization")
    improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "jpeg", "mp4"])

    frame_skip = st.slider("Frame Skip (for video)", 1, 100, 1)

    if uploaded_file is not None:
        if uploaded_file.type.startswith('video'):
            # Handle video
            video_frames = read_video_frames(uploaded_file, frame_skip)
            st.video(uploaded_file)
            output_audio_path = "output_audio.wav"
            video_to_audio(
                video_frames,
                output_audio_path,
                time=time,
                n_fft=n_fft,
                hop_length=hop_length,
                n_iter=n_iter,
                contrast_stretch=contrast_stretch,
                hist_equalize=hist_equalize,
                improve_reconstruction=improve_reconstruction,
            )
            st.success("Audio generation complete. Click the button below to download the audio.")
            audio_bytes = open(output_audio_path, "rb").read()
            st.audio(audio_bytes, format="audio/wav")
        else:
            # Handle image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Generate Audio"):
                image_np = np.array(image)

                audio = img_to_audio(
                    image_np,
                    time=time,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_iter=n_iter,
                    contrast_stretch=contrast_stretch,
                    hist_equalize=hist_equalize,
                    improve_reconstruction=improve_reconstruction,
                )

                st.audio(audio[1], format="audio/wav", sample_rate=audio[0])
                generate_waveform(audio[1], audio[0])

if __name__ == "__main__":
    main()
