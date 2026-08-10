import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Bottle Quality Detector")
st.title("Bottle Quality Detector")
st.write("Upload a bottle image to classify it as good or defective.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded bottle image", use_container_width=True)

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
