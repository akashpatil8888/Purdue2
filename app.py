import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# Configure the page
st.set_page_config(
    page_title="Bottle Quality Detector",
    layout="wide"
)

# Fixed preview box size
DISPLAY_WIDTH = 350
DISPLAY_HEIGHT = 350

# Create a counter used to reset the file uploader
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

# Page heading
st.title("Bottle Quality Detector")
st.write("Upload a bottle image to check whether it is upright or crushed.")

# Load the trained YOLO classification model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# The key changes after Clear is clicked, creating a fresh uploader
uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"],
    key=f"bottle_uploader_{st.session_state.uploader_version}"
)

if uploaded_file is not None:
    # Original full-size image used by the model
    image = Image.open(uploaded_file).convert("RGB")

    # Fixed-size display preview only.
    # The original image's proportions are preserved and empty space is light gray.
    preview = ImageOps.pad(
        image,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
        centering=(0.5, 0.5)
    )

    # Create left and right columns
    col_left, col_right = st.columns([1, 1])

    # LEFT: fixed-size image preview
    with col_left:
        st.subheader("Uploaded image")

        st.image(
            preview,
            caption="Bottle image preview",
            width=DISPLAY_WIDTH
        )

    # RIGHT: detailed inspection result
    with col_right:
        st.subheader("Inspection result")

        with st.spinner("Analyzing image..."):
            result = model(image)[0]

        # Read the top classification prediction
        class_id = result.probs.top1
        model_label = result.names[class_id]
        confidence = float(result.probs.top1conf)

        # Convert internal model labels to user-facing labels
        if model_label.lower() == "good":
            display_label = "Upright bottle"
            status = "PASSED"
            description = (
                "The bottle appears to be upright and in acceptable condition "
                "based on the model's classification."
            )
        else:
            display_label = "Crushed bottle"
            status = "REVIEW REQUIRED"
            description = (
                "The bottle appears to be crushed or defective. "
                "Please remove it from the approved bottle group for review."
            )

        # Result card
        with st.container(border=True):
            st.markdown("### Bottle condition")

            if model_label.lower() == "good":
                st.success(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

            st.markdown(f"## {display_label}")

            st.metric(
                label="Model confidence",
                value=f"{confidence:.1%}"
            )

            st.write(description)

            st.divider()

            st.caption(f"Model classification: {model_label}")
            st.caption(
                "Confidence represents how certain the model is about this prediction."
            )

        # Clear button
        if st.button("Clear results / Upload another image"):
            st.session_state.uploader_version += 1
            st.rerun()

else:
    st.info("Please upload a bottle image to begin.")
