# from PIL import Image, ImageOps
# import numpy as np
# import librosa
# import gradio as gr
# import matplotlib.pyplot as plt

# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
#     # load image
#     img = Image.fromarray(image).convert("L")
    
#     # apply preprocessing techniques
#     if contrast_stretch:
#         img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
#     if hist_equalize:
#         img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")
    
#     # calculate spectrogram size 
#     spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
#     spec = np.asarray(img.resize(spec_shape))
#     spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
#     spec = librosa.db_to_amplitude(spec)
    
#     if improve_reconstruction:
#         # Use advanced reconstruction method
#         audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
#     else:
#         # Use Griffin-Lim for reconstruction
#         audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)
    
#     return (rate, audio)

# def generate_spectrogram(audio, rate, n_fft=1024, hop_length=512, save_path='spectrogram.png'):
#     # Compute the spectrogram
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

#     # Create a figure and axis for the spectrogram
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(D, sr=rate, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')

#     # Save the spectrogram as an image
#     plt.savefig(save_path, bbox_inches='tight')


# time = gr.Number(3.0, label="Audio Time", min=1.0, max=10.0, step=0.1)
# n_fft = gr.Number(1024, label="n_fft", min=512, max=2048, step=64)
# hop_length = gr.Number(512, label="hop_length", min=256, max=1024, step=64)
# n_iter = gr.Number(64, label="n_iter", min=10, max=100, step=10)

# contrast_stretch = gr.Checkbox(False, label="Apply Contrast Stretching")
# hist_equalize = gr.Checkbox(False, label="Apply Histogram Equalization")
# improve_reconstruction = gr.Checkbox(False, label="Improve Griffin-Lim Reconstruction")

# image = gr.inputs.Image(label="Input Image")

# def main(time, n_fft, hop_length, n_iter, image, contrast_stretch, hist_equalize, improve_reconstruction):
#     return img_to_audio(image, time=time, n_fft=int(n_fft), hop_length=int(hop_length),
#                         n_iter=int(n_iter), contrast_stretch=contrast_stretch, hist_equalize=hist_equalize,
#                         improve_reconstruction=improve_reconstruction)
    

# desc = "Upload an image you would like to hear. Fine-tune parameters like time and n_iter for desired audio output."

# interface = gr.Interface(fn=main, inputs=[time, n_fft, hop_length, n_iter, image, contrast_stretch, hist_equalize, improve_reconstruction], outputs="audio", title="Improved Image Sonification", description=desc)

# interface.launch()


# from PIL import Image, ImageOps
# import numpy as np
# import librosa
# import gradio as gr
# import matplotlib.pyplot as plt

# # Function to convert image to audio
# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
#     # Load image
#     img = Image.fromarray(image).convert("L")
    
#     # Apply preprocessing techniques
#     if contrast_stretch:
#         img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
#     if hist_equalize:
#         img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")
    
#     # Calculate spectrogram size 
#     spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
#     spec = np.asarray(img.resize(spec_shape))
#     spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
#     spec = librosa.db_to_amplitude(spec)
    
#     if improve_reconstruction:
#         # Use advanced reconstruction method
#         audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
#     else:
#         # Use Griffin-Lim for reconstruction
#         audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)
    
#     return (rate, audio)

# # Function to generate and display the spectrogram
# def generate_spectrogram(audio, rate, n_fft=1024, hop_length=512, save_path='spectrogram.png'):
#     # Compute the spectrogram
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

#     # Create a figure and axis for the spectrogram
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(D, sr=rate, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')

#     # Save the spectrogram as an image
#     plt.savefig(save_path, bbox_inches='tight')

# # Gradio interface for generating audio
# time = gr.Number(3.0, label="Audio Time", min=1.0, max=10.0, step=0.1)
# n_fft = gr.Number(1024, label="n_fft", min=512, max=2048, step=64)
# hop_length = gr.Number(512, label="hop_length", min=256, max=1024, step=64)
# n_iter = gr.Number(64, label="n_iter", min=10, max=100, step=10)

# contrast_stretch = gr.Checkbox(False, label="Apply Contrast Stretching")
# hist_equalize = gr.Checkbox(False, label="Apply Histogram Equalization")
# improve_reconstruction = gr.Checkbox(False, label="Improve Griffin-Lim Reconstruction")

# image = gr.inputs.Image(label="Input Image")

# def main(time, n_fft, hop_length, n_iter, image, contrast_stretch, hist_equalize, improve_reconstruction):
#     return img_to_audio(image, time=time, n_fft=int(n_fft), hop_length=int(hop_length),
#                         n_iter=int(n_iter), contrast_stretch=contrast_stretch, hist_equalize=hist_equalize,
#                         improve_reconstruction=improve_reconstruction)

# desc = "Upload an image you would like to hear. Fine-tune parameters like time and n_iter for desired audio output."

# interface = gr.Interface(fn=main, inputs=[time, n_fft, hop_length, n_iter, image, contrast_stretch, hist_equalize, improve_reconstruction], outputs="audio", title="Improved Image Sonification", description=desc)

# # Gradio interface for displaying the spectrogram
# def show_spectrogram():
#     global audio
#     generate_spectrogram(audio, rate, n_fft=int(n_fft), hop_length=int(hop_length), save_path='spectrogram.png')
#     spectrogram_image = Image.open('spectrogram.png')
#     return spectrogram_image

# spectrogram_interface = gr.Interface(fn=show_spectrogram, inputs=None, outputs="image", title="Spectrogram", live=False)


# interface.launch()
# spectrogram_interface.launch()

# ----------------------------------------------------------------------------------------------

# import streamlit as st
# import numpy as np
# import librosa
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt

# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
#     # Load image
#     img = Image.fromarray(image).convert("L")

#     # Apply preprocessing techniques
#     if contrast_stretch:
#         img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
#     if hist_equalize:
#         img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

#     # Calculate spectrogram size
#     spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
#     spec = np.asarray(img.resize(spec_shape))
#     spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
#     spec = librosa.db_to_amplitude(spec)

#     if improve_reconstruction:
#         # Use advanced reconstruction method
#         audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
#     else:
#         # Use Griffin-Lim for reconstruction
#         audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

#     return (rate, audio)

# def generate_spectrogram(audio, rate, n_fft=1024, hop_length=512, save_path='spectrogram.png'):
#     # Compute the spectrogram
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

#     # Create a figure and axis for the spectrogram
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(D, sr=rate, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')

#     # Save the spectrogram as an image
#     plt.savefig(save_path, bbox_inches='tight')

# def main():
#     st.title("Improved Image Sonification")


#     time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
#     n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
#     hop_length = st.slider("hop_length", 256, 1024, 512, 64)
#     n_iter = st.slider("n_iter", 10, 100, 64, 10)
#     contrast_stretch = st.checkbox("Apply Contrast Stretching")
#     hist_equalize = st.checkbox("Apply Histogram Equalization")
#     improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         if st.button("Generate Audio"):
#             # Convert the Image object to a NumPy array
#             image_np = np.array(image)

#             audio = img_to_audio(
#                 image_np,
#                 time=time,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 n_iter=n_iter,
#                 contrast_stretch=contrast_stretch,
#                 hist_equalize=hist_equalize,
#                 improve_reconstruction=improve_reconstruction,
#             )

#             # Display the audio with the sample rate as metadata
#             st.audio(audio[1], format="audio/wav", sample_rate=audio[0])

# main()

# ----------------------------------------------------------------------------------------------------------------------------------
# import streamlit as st
# import numpy as np
# import librosa
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
#     # Load image
#     img = Image.fromarray(image).convert("L")

#     # Apply preprocessing techniques
#     if contrast_stretch:
#         img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
#     if hist_equalize:
#         img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

#     # Calculate spectrogram size
#     spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
#     spec = np.asarray(img.resize(spec_shape))
#     spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
#     spec = librosa.db_to_amplitude(spec)

#     if improve_reconstruction:
#         # Use advanced reconstruction method
#         audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
#     else:
#         # Use Griffin-Lim for reconstruction
#         audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

#     # Apply smoothing to make the audio more appealing
#     audio = smooth_audio(audio)

#     return (rate, audio)

# def smooth_audio(audio, sigma=1):
#     # Apply Gaussian smoothing to the audio
#     smoothed_audio = gaussian_filter(audio, sigma=sigma)
#     return smoothed_audio

# def generate_spectrogram(audio, rate, n_fft=1024, hop_length=512):
#     # Compute the spectrogram
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

#     # Display the spectrogram as an image
#     st.image(D, caption="Spectrogram", use_column_width=True)

# def main():
#     st.title("Improved Image Sonification")

#     time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
#     n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
#     hop_length = st.slider("hop_length", 256, 1024, 512, 64)
#     n_iter = st.slider("n_iter", 10, 100, 64, 10)
#     contrast_stretch = st.checkbox("Apply Contrast Stretching")
#     hist_equalize = st.checkbox("Apply Histogram Equalization")
#     improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         if st.button("Generate Audio"):
#             # Convert the Image object to a NumPy array
#             image_np = np.array(image)

#             audio = img_to_audio(
#                 image_np,
#                 time=time,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 n_iter=n_iter,
#                 contrast_stretch=contrast_stretch,
#                 hist_equalize=hist_equalize,
#                 improve_reconstruction=improve_reconstruction,
#             )

#             # Display the audio with the sample rate as metadata
#             st.audio(audio[1], format="audio/wav", sample_rate=audio[0])

#             # Generate and display the spectrogram
#             generate_spectrogram(audio[1], audio[0], n_fft=n_fft, hop_length=hop_length)

# if __name__ == "__main__":
#     main()
# import streamlit as st
# import numpy as np
# import librosa
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# import soundfile as sf
# from scipy.signal import butter, lfilter, sosfreqz, sosfilt
# import math

# def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False, loudness_factor=6.0):
#     # Load image
#     img = Image.fromarray(image).convert("L")

#     # Apply preprocessing techniques
#     if contrast_stretch:
#         img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
#     if hist_equalize:
#         img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")

#     # Calculate spectrogram size
#     spec_shape = (int(librosa.time_to_frames(1.0, sr=rate, hop_length=hop_length, n_fft=n_fft) * time), n_fft)
#     spec = np.asarray(img.resize(spec_shape))
#     spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
#     spec = librosa.db_to_amplitude(spec)

#     if improve_reconstruction:
#         # Use advanced reconstruction method
#         audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
#     else:
#         # Use Griffin-Lim for reconstruction
#         audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

#     # Apply smoothing to make the audio more appealing
#     audio = smooth_audio(audio)

#     # Amplify the audio signal for more intensity
#     audio *= loudness_factor

#     # Apply a low-pass filter for a more musical tone (adjust cutoff frequency as needed)
#     cutoff_frequency = 5000  # Adjust this value to emphasize lower frequencies
#     audio = apply_low_pass_filter(audio, rate, cutoff_frequency)

#     # Add harmonics to make it more musical
#     harmonics = generate_harmonics(audio, rate)
#     audio += harmonics

#     # Apply reverb for a more pleasing sound
#     audio = apply_reverb(audio, rate)

#     # Adjust the envelope for a smoother sound
#     audio = adjust_envelope(audio, rate)

#     # Save the audio as a WAV file
#     sf.write("output_audio.wav", audio, rate)

#     return (rate, audio)

# def smooth_audio(audio, sigma=2):
#     # Apply Gaussian smoothing to the audio
#     smoothed_audio = gaussian_filter(audio, sigma=sigma)
#     return smoothed_audio

# def apply_low_pass_filter(audio, rate, cutoff_frequency):
#     nyquist_frequency = 0.5 * rate
#     normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
#     sos = butter(10, normalized_cutoff_frequency, btype='low', output='sos')
#     filtered_audio = sosfilt(sos, audio)
#     return filtered_audio

# def generate_harmonics(audio, rate, num_harmonics=5, harmonic_ratio=2.0):
#     harmonics = np.zeros_like(audio)
#     for i in range(1, num_harmonics + 1):
#         harmonic = audio.copy()
#         for j in range(len(harmonic)):
#             harmonic[j] *= math.sin(2 * math.pi * (i / harmonic_ratio) * (j / rate))
#         harmonics += harmonic
#     return harmonics


# def apply_reverb(audio, rate, reverb_duration=0.2, reverb_amplitude=0.2):
#     reverb_samples = int(reverb_duration * rate)
#     reverb = np.random.randn(reverb_samples) * reverb_amplitude
#     audio_with_reverb = np.concatenate((audio, reverb))
#     return audio_with_reverb

# def adjust_envelope(audio, rate):
#     # Apply an envelope to make the sound smoother
#     envelope = np.linspace(0.1, 1, len(audio))
#     audio_with_envelope = audio * envelope
#     return audio_with_envelope

# def generate_waveform(audio, rate):
#     # Generate and display the waveform graph
#     plt.figure(figsize=(10, 4))
#     plt.plot(np.arange(len(audio)) / rate, audio)
#     plt.title('Sound Waveform')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     st.pyplot()

# def main():
#     st.title("Waveform Graph and Audio")

#     time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
#     n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
#     hop_length = st.slider("hop_length", 256, 1024, 512, 64)
#     n_iter = st.slider("n_iter", 10, 100, 64, 10)
#     contrast_stretch = st.checkbox("Apply Contrast Stretching")
#     hist_equalize = st.checkbox("Apply Histogram Equalization")
#     improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
#     loudness_factor = st.slider("Loudness Factor", 1.0, 10.0, 6.0, 0.1)
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         if st.button("Generate Graph and Audio"):
#             # Convert the Image object to a NumPy array
#             image_np = np.array(image)

#             audio = img_to_audio(
#                 image_np,
#                 time=time,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 n_iter=n_iter,
#                 contrast_stretch=contrast_stretch,
#                 hist_equalize=hist_equalize,
#                 improve_reconstruction=improve_reconstruction,
#                 loudness_factor=loudness_factor,
#             )

#             # Generate and display the waveform graph
#             generate_waveform(audio[1], audio[0])

#             # Display the audio
#             st.audio("output_audio.wav", format="audio/wav")

# if __name__ == "__main__":
#     main()
import streamlit as st
import numpy as np
import librosa
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
    spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
    spec = librosa.db_to_amplitude(spec)

    if improve_reconstruction:
        # Use advanced reconstruction method
        audio = librosa.effects.preemphasis(librosa.feature.inverse.mel_to_audio(spec))
    else:
        # Use Griffin-Lim for reconstruction
        audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)

    # Apply smoothing to make the audio more appealing
    audio = smooth_audio(audio)

    return (rate, audio)

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

def main():
    st.title("Improved Image Sonification")

    time = st.slider("Audio Time (seconds)", 1.0, 50.0, 3.0, 0.1)
    n_fft = st.slider("n_fft", 512, 2048, 1024, 64)
    hop_length = st.slider("hop_length", 256, 1024, 512, 64)
    n_iter = st.slider("n_iter", 10, 100, 64, 10)
    contrast_stretch = st.checkbox("Apply Contrast Stretching")
    hist_equalize = st.checkbox("Apply Histogram Equalization")
    improve_reconstruction = st.checkbox("Improve Griffin-Lim Reconstruction")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
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
