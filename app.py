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

# Page heading
st.title("Bottle Quality Detector")
st.write("Upload a bottle image to check whether it is upright or crushed.")

# Load the trained YOLO classification model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Image uploader
uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Original image used by the model
    image = Image.open(uploaded_file).convert("RGB")

    # Display-only preview:
    # Every image appears inside the same 350 x 350 pixel box.
    # Aspect ratio is preserved, with light-gray padding if needed.
    preview = ImageOps.pad(
        image,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
        centering=(0.5, 0.5)
    )

    # Create left and right columns
    col_left, col_right = st.columns([1, 1])

    # LEFT SIDE: Fixed-size uploaded image preview
    with col_left:
        st.subheader("Uploaded image")

        st.image(
            preview,
            caption="Bottle image preview",
            width=DISPLAY_WIDTH
        )

    # RIGHT SIDE: Detailed prediction result
    with col_right:
        st.subheader("Inspection result")

        with st.spinner("Analyzing image..."):
            result = model(image)[0]

        # Get top prediction from the YOLO classification model
        class_id = result.probs.top1
        model_label = result.names[class_id]
        confidence = float(result.probs.top1conf)

        # Display user-friendly labels and descriptions
        if model_label.lower() == "good":
            display_label = "Upright bottle"
            status = "PASSED"
            description = (
                "The bottle appears to be upright and in acceptable condition "
                "based on the model's classification."
            )
            message_type = "success"
        else:
            display_label = "Crushed bottle"
            status = "REVIEW REQUIRED"
            description = (
                "The bottle appears to be crushed or defective. "
                "Please remove it from the approved bottle group for review."
            )
            message_type = "error"

        # Group all result information in one visual card
        with st.container(border=True):
            st.markdown("### Bottle condition")

            if message_type == "success":
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
            st.caption("Confidence represents how certain the model is about this prediction.")

else:
    st.info("Please upload a bottle image to begin.")
