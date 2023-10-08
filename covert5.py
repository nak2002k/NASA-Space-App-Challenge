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
from pywavefront import Wavefront
import pywavefront

def img_to_audio(image, time=3.0, rate=44100, n_fft=2048, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
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
        audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
    else:
        # Use Griffin-Lim for reconstruction
        audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

    # Apply smoothing to make the audio more appealing
    audio = smooth_audio(audio)

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

def video_to_audio(video_frames, output_audio_path, time=3.0, rate=44100, n_fft=2048, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
    audio_frames = []
    video_frame_rate = len(video_frames) / time  # Calculate the frame rate of the video

    for i, frame in enumerate(video_frames):
        # Calculate the corresponding time for the audio segment
        audio_time = i / video_frame_rate

        # Ensure that we generate audio only within the specified time
        if audio_time >= time:
            break

        audio = img_to_audio(frame, time, rate, n_fft, n_iter, hop_length, contrast_stretch, hist_equalize, improve_reconstruction)
        audio_frames.append(audio[1])

    audio_frames = np.concatenate(audio_frames)

    # Save the resulting audio as a WAV file
    sf.write(output_audio_path, audio_frames, rate)


def obj_to_audio(obj_file_path, time=3.0, rate=22050, sigma=1):
    try:
        # Read and process the .obj file
        obj = Wavefront(obj_file_path)
        # Define the range for mapping x-coordinate (min_x and max_x) to pitch (min_pitch and max_pitch)
        min_x = min(vertices, key=lambda x: x[0])[0]  # Find the minimum x-coordinate in your vertices
        max_x = max(vertices, key=lambda x: x[0])[0]  # Find the maximum x-coordinate in your vertices
        min_pitch = 100  # Minimum pitch value
        max_pitch = 1000  # Maximum pitch value

        # Define the range for mapping y-coordinate (min_y and max_y) to volume (min_volume and max_volume)
        min_y = min(vertices, key=lambda x: x[1])[1]  # Find the minimum y-coordinate in your vertices
        max_y = max(vertices, key=lambda x: x[1])[1]  # Find the maximum y-coordinate in your vertices
        min_volume = -20  # Minimum volume in dB (e.g., -20 dB)
        max_volume = 0  # Maximum volume in dB (e.g., 0 dB)
        # Extract vertex positions
        vertices = obj.vertices

        audio_segments = []  # Store audio segments for each vertex

        for vertex in vertices:
            if len(vertex) != 3:
                continue

            x, y, z = vertex

            # Map vertex positions to audio parameters (example: pitch and volume)
            pitch = map_to_range(x, min_x, max_x, min_pitch, max_pitch)
            volume = map_to_range(z, min_x, max_x, min_volume, max_volume)

            # Create an audio segment for this vertex
            vertex_audio = AudioSegment.silent(duration=int(time * 1000))  # Duration in milliseconds
            vertex_audio = vertex_audio + AudioSegment.silent(duration=100)  # A small gap between vertex sounds
            vertex_audio = vertex_audio + create_audio_from_parameters(time, rate, pitch, volume)

            audio_segments.append(vertex_audio)

        # Combine all audio segments into a single audio
        audio = AudioSegment.silent(duration=0)
        for segment in audio_segments:
            audio += segment

        # Apply Gaussian smoothing to the audio
        audio = audio.low_pass_filter(sigma * 1000)  # Sigma in Hz

        # Generate a unique output audio path
        audio_path = f"output_audio_{int(time)}s.wav"
        audio.export(audio_path, format="wav")

        return audio_path

    except Exception as e:
        st.error(f"Error processing OBJ file: {str(e)}")
        return None

# Helper function to map values from one range to another
def map_to_range(value, from_min, from_max, to_min, to_max):
    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def create_audio_from_parameters(time, rate, pitch, volume):
    # Define audio properties
    duration_ms = int(time * 1000)  # Duration in milliseconds
    sample_rate = rate  # Sample rate in Hz
    num_samples = int(duration_ms * sample_rate / 1000)

    # Create time values for the audio
    t = np.linspace(0, time, num_samples, endpoint=False)

    # Generate audio waveform based on pitch and volume
    frequency = 440.0 * 2**(pitch / 12.0)  # Calculate frequency from pitch (assuming A440 reference)
    amplitude = 0.5 * volume  # Adjust volume

    # Generate a simple sine wave as an example
    audio_data = amplitude * np.sin(2 * np.pi * frequency * t)

    # Convert the audio data to a PyDub AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),  # Audio data as bytes
        frame_rate=sample_rate,  # Sample rate
        sample_width=audio_data.dtype.itemsize,  # Sample width in bytes
        channels=1  # Mono audio
    )

    return audio_segment




def main():
    st.title("Improved Image, Video, and 3D Object Sonification")

    time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
    n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
    hop_length = st.slider("hop_length", 256, 1024, 512, 64)
    n_iter = st.slider("n_iter", 10, 100, 64, 10)
    contrast_stretch = st.checkbox("Apply Contrast Stretching")
    hist_equalize = st.checkbox("Apply Histogram Equalization")
    improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
    uploaded_file = st.file_uploader("Upload a 3D image or video", type=["jpg", "png", "jpeg", "mp4", "obj"])
    
    frame_skip = st.slider("Frame Skip", 1, 100, 1)

    if uploaded_file is not None:
        # Check if the uploaded file is an image, video, or 3D object
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
        elif uploaded_file.name.endswith('.obj'):
            # Handle 3D object
            st.info("Processing the 3D object...")

            # Save the uploaded .obj file temporarily
            with open("temp_obj.obj", "wb") as temp_obj_file:
                temp_obj_file.write(uploaded_file.read())

            # Perform sonification
            audio_path = obj_to_audio("temp_obj.obj")

            st.success("Sonification complete. Click the button below to play the audio.")
            st.audio(audio_path, format="audio/wav")

            # Remove the temporary .obj file
            os.remove("temp_obj.obj")
        else:
            # Handle image
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Generate Audio"):
                    # Convert the Image object to a NumPy array
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

                    # Display the audio with the sample rate as metadata
                    st.audio(audio[1], format="audio/wav", sample_rate=audio[0])

                    # Generate and display the waveform plot
                    generate_waveform(audio[1], audio[0])

if __name__ == "__main__":
    main()
