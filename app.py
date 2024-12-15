import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load your model (replace 'best.pt' with your trained model file)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# App title and description
st.title("Weapon Recognition App")
st.write("Upload an image or use your webcam for weapon detection.")

# Sidebar for input options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Choose an input source:", ("Upload Image", "Webcam"))

# RTC Configuration with free TURN servers
RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": ["turn:relay.metered.ca:80"],
            "username": "open",
            "credential": "open"
        },
        {
            "urls": ["turn:relay.metered.ca:443"],
            "username": "open",
            "credential": "open"
        },
        {
            "urls": ["turn:relay.metered.ca:5349"],
            "username": "open",
            "credential": "open"
        }
    ]
}

# Detection function
def detect_objects(image):
    results = model(image)
    annotated_image = results[0].plot()
    return annotated_image

# Handle uploaded images
if input_type == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")
        
        # Convert image to OpenCV format
        image_np = np.array(image)
        processed_image = detect_objects(image_np)
        
        # Display the results
        st.image(processed_image, caption="Detected Objects", use_column_width=True)

# Webcam processing using streamlit-webrtc
elif input_type == "Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            # Convert the frame to an OpenCV image
            img = frame.to_ndarray(format="bgr24")
            processed_frame = detect_objects(img)
            return processed_frame

    # Add the RTC configuration to the webrtc_streamer function
    webrtc_streamer(
        key="weapon-detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,  # Pass the RTC configuration here
    )
