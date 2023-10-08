
import streamlit as st
import covert5  # Import the module where your main script logic resides
from pywavefront import Wavefront

# Set the page title and favicon
st.set_page_config(
    page_title="Sonification of Sounds",
    page_icon="🎵",
    layout="wide",
)

# Define the background image CSS
background_image_style = """
<style>
body {
    background-image: url("./BackgroundImg.jpg");  # Update with the correct image URL
    background-size: cover;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center; /* Center-align text */
}
</style>
"""

# Display the background image and title
st.markdown(background_image_style, unsafe_allow_html=True)
st.title("Sonification of Sounds")
st.image("https://c4.wallpaperflare.com/wallpaper/349/372/16/abstract-space-nebula-space-art-wallpaper-preview.jpg", use_column_width=True)

# Add a button to start the main script
if st.button("Start Sonifying"):
    # Call the main script logic from the `covert5` module
    covert5.main()  # Replace `main()` with the actual function name from your `covert5` module
