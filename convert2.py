from PIL import Image, ImageOps
import numpy as np
import librosa
import gradio as gr

def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False, improve_reconstruction=False):
    # load image
    img = Image.fromarray(image).convert("L")
    
    # apply preprocessing techniques
    if contrast_stretch:
        img = Image.fromarray(np.uint8(255 * (np.asarray(img) - np.min(img)) / (np.max(img) - np.min(img))))
    if hist_equalize:
        img = ImageOps.equalize(ImageOps.autocontrast(img)).convert("L")
    
    # calculate spectrogram size 
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
    
    return (rate, audio)

time = gr.Number(3.0, label="Audio Time", min=1.0, max=10.0, step=0.1)
n_fft = gr.Number(1024, label="n_fft", min=512, max=2048, step=64)
hop_length = gr.Number(512, label="hop_length", min=256, max=1024, step=64)
n_iter = gr.Number(64, label="n_iter", min=10, max=100, step=10)

contrast_stretch = gr.Checkbox(False, label="Apply Contrast Stretching")
hist_equalize = gr.Checkbox(False, label="Apply Histogram Equalization")
improve_reconstruction = gr.Checkbox(False, label="Improve Griffin-Lim Reconstruction")

image = gr.inputs.Image(label="Input Image")

def main(time, n_fft, hop_length, n_iter, image, contrast_stretch, hist_equalize, improve_reconstruction):
    return img_to_audio(image, time=time, n_fft=int(n_fft), hop_length=int(hop_length),
                        n_iter=int(n_iter), contrast_stretch=contrast_stretch, hist_equalize=hist_equalize,
                        improve_reconstruction=improve_reconstruction)

desc = "Upload an image you would like to hear. Fine-tune parameters like time and n_iter for desired audio output."

interface = gr.Interface(fn=main, inputs=[image,time, n_fft, hop_length, n_iter, contrast_stretch, hist_equalize, improve_reconstruction], outputs="audio", title="Improved Image Sonification", description=desc)

interface.launch()