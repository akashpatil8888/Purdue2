import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bottle Quality Detector · Department of Food Science, Purdue University",
    layout="wide"
)

# -------------------------------------------------------------------
# Purdue-branded styling (colors, typography, layout)
# -------------------------------------------------------------------
Purdue_GOLD = "#CFB991"   # Boilermaker Gold
Purdue_BLACK = "#000000"  # Black
Purdue_STEEL = "#555960"  # Supporting neutral gray
Purdue_DUST = "#EBD99F"   # Soft gold/cream accent

st.markdown(
    f"""
    <style>
    /* App background and base font */
    .stApp {{
        background-color: #FFFFFF;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont,
                     "Helvetica Neue", Arial, sans-serif;
        color: {Purdue_BLACK};
    }}

    /* Headings with more academic feel */
    h1, h2, h3, h4 {{
        font-family: "Georgia", "Times New Roman", serif;
        color: {Purdue_BLACK};
    }}

    /* Custom Purdue header bar */
    .purdue-header {{
        padding: 1.2rem 1.5rem;
        background: linear-gradient(90deg, {Purdue_BLACK} 0%, {Purdue_STEEL} 35%, {Purdue_GOLD} 100%);
        color: #FFFFFF;
        border-radius: 0 0 10px 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25);
    }}

    .purdue-header-title {{
        font-size: 1.0rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {Purdue_DUST};
        margin-bottom: 0.35rem;
    }}

    .purdue-header-main {{
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.1rem 0;
    }}

    .purdue-header-sub {{
        font-size: 0.95rem;
        opacity: 0.92;
    }}

    /* Uploaded image preview */
    .uploaded-preview img {{
        border-radius: 8px;
        border: 1px solid {Purdue_GOLD};
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
        background-color: #F9F6F0;
    }}

    /* Result card styling */
    .result-card {{
        background-color: #F8F5F0;
        border-radius: 10px;
        border: 1px solid {Purdue_GOLD};
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }}

    .result-card h3 {{
        margin-top: 0;
        margin-bottom: 0.6rem;
    }}

    .result-status-pass {{
        color: #006400;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}

    .result-status-review {{
        color: #8B0000;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}

    .result-label {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }}

    .result-meta {{
        font-size: 0.9rem;
        color: {Purdue_STEEL};
        margin-top: 0.6rem;
    }}

    /* Footer area */
    .purdue-footer {{
        margin-top: 3rem;
        padding-top: 1.0rem;
        border-top: 1px solid #9D9795;
        font-size: 0.85rem;
        color: {Purdue_STEEL};
    }}

    .purdue-footer strong {{
        color: {Purdue_BLACK};
    }}

    .purdue-footer a {{
        color: #8E6F3E;
        text-decoration: none;
        font-weight: 500;
    }}

    .purdue-footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Persistent state for clearing the uploader
# -------------------------------------------------------------------
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

# -------------------------------------------------------------------
# Purdue header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="purdue-header">
        <div class="purdue-header-title">
            Department of Food Science · Purdue University
        </div>
        <div class="purdue-header-main">
            Bottle Quality Detector
        </div>
        <div class="purdue-header-sub">
            College of Agriculture · Digital Quality Assessment Prototype
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a bottle image to classify its condition as an upright or crushed bottle, "
    "using a trained image classification model."
)

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------------------------------------------
# Fixed preview box size
# -------------------------------------------------------------------
DISPLAY_WIDTH = 350
DISPLAY_HEIGHT = 350

# -------------------------------------------------------------------
# Image uploader (resettable via session_state)
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a bottle photo",
    type=["jpg", "jpeg", "png"],
    key=f"bottle_uploader_{st.session_state.uploader_version}"
)

if uploaded_file is not None:
    # Original image for the model
    image = Image.open(uploaded_file).convert("RGB")

    # Fixed-size preview (visual only)
    preview = ImageOps.pad(
        image,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
        centering=(0.5, 0.5)
    )

    # Layout: image left, result right
    col_left, col_right = st.columns([1, 1])

    # ----------------------------------------------------------------
    # LEFT: Uploaded image preview
    # ----------------------------------------------------------------
    with col_left:
        st.subheader("Uploaded image")

        st.markdown('<div class="uploaded-preview">', unsafe_allow_html=True)
        st.image(
            preview,
            caption="Bottle image preview",
            width=DISPLAY_WIDTH
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # RIGHT: Inspection result
    # ----------------------------------------------------------------
    with col_right:
        st.subheader("Inspection result")

        with st.spinner("Analyzing image..."):
            result = model(image)[0]

        class_id = result.probs.top1
        model_label = result.names[class_id]
        confidence = float(result.probs.top1conf)

        # Map internal model labels to user-facing Purdue wording
        if model_label.lower() == "good":
            display_label = "Upright bottle"
            status_text = "PASSED – visual inspection"
            description = (
                "The bottle appears upright and structurally acceptable based on the model's classification. "
                "No obvious signs of crushing or major deformation have been detected."
            )
            status_class = "result-status-pass"
        else:
            display_label = "Crushed bottle"
            status_text = "REVIEW REQUIRED – potential defect"
            description = (
                "The bottle appears to be crushed or defective. "
                "Please route this unit for manual inspection and remove it from approved product flow."
            )
            status_class = "result-status-review"

        # Result card with Purdue styling
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label">{display_label}</div>', unsafe_allow_html=True)

        st.metric(
            label="Model confidence",
            value=f"{confidence:.1%}"
        )

        st.write(description)

        st.markdown(
            f"""
            <div class="result-meta">
                Model classification label: <strong>{model_label}</strong><br/>
                Confidence reflects how certain the model is about this prediction, based on the trained dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)  # close result-card

        # Clear button to reset uploader and results
        if st.button("Clear results / Upload another image"):
            st.session_state.uploader_version += 1
            st.rerun()

else:
    st.info("Please upload a bottle image to begin the inspection.")

# -------------------------------------------------------------------
# Footer suitable for university hosting
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="purdue-footer">
        <p>
            <strong>Department of Food Science, Purdue University</strong><br/>
            745 Agriculture Mall Drive · West Lafayette, IN 47907 · USA<br/>
            For program information, visit the
            <a href="https://ag.purdue.edu/department/foodsci/" target="_blank">
                Department of Food Science website
            </a>.
        </p>
        
    </div>
    """,
    unsafe_allow_html=True,
)
