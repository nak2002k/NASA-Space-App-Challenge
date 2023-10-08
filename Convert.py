import streamlit as st
import numpy as np
import librosa
from PIL import Image, ImageOps
import cv2
import tempfile
from scipy.io.wavfile import write as write_wav

def img_to_audio(image, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
    img = Image.fromarray(image).convert("L")

    if contrast_stretch:
        img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
    if hist_equalize:
        img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

    img_np = np.array(img)

    if img_np.shape[0] <= 0 or img_np.shape[1] <= 0:
        st.error("Invalid image size. Please upload a valid image.")
        return None

    img_resized = cv2.resize(img_np, (n_fft, int(rate)))

    spec = np.interp(img_resized, (img_resized.min(), img_resized.max()), (-50, 30))
    spec = librosa.db_to_amplitude(spec)

    if improve_reconstruction:
        audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
    else:
        audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

    return rate, audio

def main():
    st.title("Improved Video Sonification")

    rate = st.slider("Audio Sampling Rate", 22050, 44100, 22050)
    n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
    n_iter = st.slider("n_iter", 10, 100, 64, 10)
    contrast_stretch = st.checkbox("Apply Contrast Stretching")
    hist_equalize = st.checkbox("Apply Histogram Equalization")
    improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
    skip_frames = 100  # Set the number of frames to skip
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        st.warning("Processing may take a while. Please be patient.")

        # Save the video to a local temporary file
        video_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(video_temp_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(video_temp_path)

        audio_frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if i % skip_frames != 0:
                continue

            audio = img_to_audio(frame, rate=rate, n_fft=n_fft, n_iter=n_iter, contrast_stretch=contrast_stretch, hist_equalize=hist_equalize, improve_reconstruction=improve_reconstruction)
            if audio is not None:
                audio_frames.append(audio[1])

        cap.release()

        if audio_frames:
            audio_frames = np.concatenate(audio_frames)

            # Save the audio as a temporary file
            audio_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            write_wav(audio_temp_path, rate, audio_frames.astype(np.int16))

            st.audio(audio_temp_path, format="audio/wav", start_time=0, sample_rate=rate)

            # Cleanup temporary files
            st.info("Processing completed.")
            st.success("Audio generated successfully.")
            st.balloons()
            st.markdown(f"**[Download Audio File]({audio_temp_path})**")

if __name__ == "__main__":
    main()
