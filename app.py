import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Configure the page
st.set_page_config(
    page_title="Bottle Quality Detector",
    layout="wide"
)

# Page heading
st.title("Bottle Quality Detector")
st.write("Upload a bottle image to classify it as good or defective.")

# Load the trained YOLO classification model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload widget
uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the original image for model analysis
    image = Image.open(uploaded_file).convert("RGB")

    # Create left and right sections
    col_left, col_right = st.columns(2)

    # LEFT: limited-size image preview
    with col_left:
        st.subheader("Uploaded image")

        # Make a display-only copy and restrict its size.
        # This does not affect the original image used by the model.
        preview = image.copy()
        preview.thumbnail((350, 350))

        st.image(
            preview,
            caption="Bottle image",
            width=350
        )

    # RIGHT: classification result
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
