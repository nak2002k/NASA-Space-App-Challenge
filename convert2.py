from PIL import Image, ImageOps
import numpy as np
import librosa
import gradio as gr

def img_to_audio(image, time=3.0, rate=22050, n_fft=1024, n_iter=64, hop_length=512, contrast_stretch=False, hist_equalize=False):
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
    print(spec.shape)
    spec = np.interp(spec, (spec.min(), spec.max()), (-50, 30))
    spec = librosa.db_to_amplitude(spec)
    audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop_length)
    return (rate, audio)

time = gr.Number(3.0, label="audio time")
image = gr.Image(label="image to sonify")
n_fft = gr.Number(1024, label="n_fft")

# hop_length is a parameter used in the img_to_audio function to determine the number of samples between successive frames of a spectrogram.
# It is used to control the trade-off between time and frequency resolution in the spectrogram.
hop_length = gr.Number(512, label="hop_length")

# contrast_stretch refers to a technique that stretches the contrast of the image to improve its dynamic range. 
# This can help to bring out details in the image that might otherwise be difficult to see.
contrast_stretch = gr.Checkbox(False, label="Apply contrast stretching")

# hist_equalize refers to a technique that equalizes the histogram of the image,
# which can help to bring out details in the image that might otherwise be difficult to see.
hist_equalize = gr.Checkbox(False, label="Apply histogram equalization")

def main(image, time, n_fft, hop_length, contrast_stretch, hist_equalize):
    return img_to_audio(image, time=time, n_fft=int(n_fft), hop_length=int(hop_length), contrast_stretch=contrast_stretch, hist_equalize=hist_equalize)

desc = "Upload an image you would like to hear."

interface = gr.Interface(fn=main, inputs=[image, time, n_fft, hop_length, contrast_stretch, hist_equalize], outputs="audio", title="Improved Image Sonification", description=desc)

interface.launch()