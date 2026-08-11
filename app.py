import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# Configure the page
st.set_page_config(
    page_title="Bottle Quality Detector",
    layout="wide"
)

st.title("Bottle Quality Detector")
st.write("Upload a bottle image to classify it as good or defective.")

# Fixed display dimensions for every uploaded image
DISPLAY_WIDTH = 350
DISPLAY_HEIGHT = 350

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Original image used for prediction
    image = Image.open(uploaded_file).convert("RGB")

    # Make a display-only image that is always exactly 350 x 350 pixels.
    # The image keeps its aspect ratio; blank space is filled with light gray.
    preview = ImageOps.pad(
        image,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
        centering=(0.5, 0.5)
    )

    # Side-by-side layout
    col_left, col_right = st.columns(2)

    # Left side: fixed-size preview
    with col_left:
        st.subheader("Uploaded image")
        st.image(
            preview,
            caption="Bottle image",
            width=DISPLAY_WIDTH
        )

    # Right side: model result
    with col_right:
        st.subheader("Detection result")

        with st.spinner("Analyzing image..."):
            result = model(image)[0]

        class_id = result.probs.top1
        label = result.names[class_id]
        confidence = float(result.probs.top1conf)

        if label.lower() == "good":
            st.success(f"GOOD bottle — confidence: {confidence:.1%}")
        else:
            st.error(f"DEFECTIVE bottle — confidence: {confidence:.1%}")

        st.caption(f"Model prediction: {label}")

else:
    st.info("Please upload a bottle image to begin.")
