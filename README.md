# 🎶 Improved Image, Video, and 3D Object Sonification 🖼️🎬🔊

This Python script, integrated with the Streamlit framework, provides an interactive web application to transform visual data from images, videos, and 3D object files into mesmerizing audible soundscapes. The process, known as sonification, entails converting visual data into meaningful audio representations, providing a unique and immersive auditory experience. 🌐🔊

## Prerequisites

Before executing the script, ensure the presence of the following Python libraries:

- `streamlit`
- `numpy`
- `librosa`
- `PIL` (Pillow)
- `matplotlib`
- `scipy`
- `opencv-python`
- `soundfile`
- `pydub`
- `pywavefront`

Install these dependencies using the command:

```bash
pip install streamlit numpy librosa pillow matplotlib scipy opencv-python soundfile pydub pywavefront
```

## How to Run

1. Clone the repository
2. Change into the project directory:
3. Install the dependencies and Run the script:

   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your web browser to engage with the application.

## Usage

1. **Audio Time Slider:**
   - Adjust the duration of the generated audio using the dedicated slider.

2. **Spectrogram Parameters:**
   - Fine-tune the parameters (`n_fft`, `hop_length`, `n_iter`) to control the spectrogram generation process.

3. **Preprocessing Options:**
   - Enable or disable preprocessing techniques (`Apply Contrast Stretching`, `Apply Histogram Equalization`, `Improve Griffin-Lim Reconstruction`) to tailor the transformation of input data.

4. **File Uploader:**
   - Upload images (JPEG, PNG), videos (MP4), or 3D object files (OBJ).

5. **Frame Skip Slider:**
   - If uploading a video, adjust the slider to skip frames during processing.

6. **Generate Audio Button:**
   - For images, click this button to convert the image to audio and visualize the waveform.
   - For videos, the audio is generated from individual video frames, and the resulting audio file can be downloaded.
   - For 3D objects, the script processes the object and plays the resulting audio.

7. **Download Audio:**
   - After audio generation, a button appears to download the audio file.

8. **Play Audio:**
   - Click the play button next to the 3D object sonification result to listen to the audio.

## Methodology and Parameters

The script employs various methods and parameters to enhance the sonification process:

- **Contrast Stretching:**
  - Applied through the checkbox `Apply Contrast Stretching`.
  - Enhances the dynamic range of pixel values in the image, ensuring a more perceptually appealing audio representation.

- **Histogram Equalization:**
  - Enabled by selecting `Apply Histogram Equalization`.
  - Improves the visibility of details in an image by redistributing pixel values across a broader range.

- **Griffin-Lim Reconstruction:**
  - Controlled by the checkbox `Improve Griffin-Lim Reconstruction`.
  - Utilizes the Griffin-Lim algorithm for audio reconstruction, enhancing the fidelity of the generated audio from the spectrogram.

- **Gaussian Smoothing:**
  - Implemented through the `smooth_audio` function.
  - Applies Gaussian smoothing to the audio, resulting in a more polished and aesthetically pleasing auditory experience.

- **Waveform Generation:**
  - Utilizes the `generate_waveform` function to plot and display the audio waveform.
  - Offers a visual representation of the amplitude variations over time in the generated audio.

These techniques, coupled with adjustable parameters such as spectrogram size, frame skip, and preprocessing options, empower users to tailor the sonification process to their specific preferences and achieve diverse and captivating audio outcomes. 🎨🔊

Feel free to explore the potential of this application to transform visual stimuli into a rich and immersive auditory experience. 🚀🔊


### Note:- 
- This project was part of a 48hr non-stop hackathon so judge us accordingly 😅
- All of the project implementation details are not added in the Readme here will add as early as possible
