# ==========================================================
# AI Powered PCB Inspection System
# Streamlit Dashboard
# ==========================================================

import os
import sys
import time
import tempfile
from pathlib import Path

import cv2
import streamlit as st

# ==========================================================
# Add Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================================
# Streamlit Components
# ==========================================================

from src.streamlit.components.header import show_header
from src.streamlit.components.sidebar import show_sidebar
from src.streamlit.components.image_view import show_images
from src.streamlit.components.metrics import show_metrics
from src.streamlit.components.tables import show_detection_table
from src.streamlit.components.charts import show_chart
from src.streamlit.components.recommendations import show_recommendations
from src.streamlit.components.footer import show_footer

# ==========================================================
# Pipeline
# ==========================================================

from src.pipeline.pipeline import PCBInspectionPipeline

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Powered PCB Inspection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CSS
# ==========================================================

css_file = Path(__file__).parent / "styles.css"

if css_file.exists():

    with open(css_file, "r", encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================================
# Load YOLO Model Only Once
# ==========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best.pt"
)

@st.cache_resource
def load_pipeline():

    return PCBInspectionPipeline(str(MODEL_PATH))

pipeline = load_pipeline()

# ==========================================================
# Header
# ==========================================================

show_header()

# ==========================================================
# Sidebar
# ==========================================================

uploaded_file, confidence = show_sidebar()

# ==========================================================
# Dashboard Title
# ==========================================================

st.markdown(
    """
    <div class="section-title">
        PCB Inspection Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# Waiting for Upload
# ==========================================================

if uploaded_file is None:

    st.info("👈 Upload a PCB image from the sidebar to begin inspection.")

    show_footer()

    st.stop()

# ==========================================================
# Save Uploaded Image
# ==========================================================

suffix = Path(uploaded_file.name).suffix

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
) as tmp:

    tmp.write(uploaded_file.read())

    temp_image_path = tmp.name

st.success(f"Image Uploaded : {uploaded_file.name}")

# ==========================================================
# Run AI Inspection
# ==========================================================

start_time = time.time()

try:

    with st.spinner("Analyzing PCB using YOLOv8..."):

        output = pipeline.run(
            image_path=temp_image_path,
            confidence=confidence
        )

except Exception as e:

    st.error("Pipeline Execution Failed")

    st.exception(e)

    os.remove(temp_image_path)

    st.stop()

processing_time = time.time() - start_time

# ==========================================================
# Convert OpenCV Image to RGB
# ==========================================================

detected_image = cv2.cvtColor(
    output["image"],
    cv2.COLOR_BGR2RGB
)

# ==========================================================
# Display Processing Time
# ==========================================================

st.caption(
    f"⏱ Processing Time : {processing_time:.2f} seconds"
)
# ==========================================================
# Display Images
# ==========================================================

show_images(
    original=uploaded_file,
    detected=detected_image
)

st.divider()

# ==========================================================
# PCB Health Metrics
# ==========================================================

show_metrics(
    output["health"]
)

st.divider()

# ==========================================================
# Defect Distribution Chart
# ==========================================================

show_chart(
    output["health"]
)

st.divider()

# ==========================================================
# Detection Table
# ==========================================================

show_detection_table(
    output["detections"]
)

st.divider()

# ==========================================================
# Recommendations
# ==========================================================

show_recommendations(
    output["recommendations"]
)

st.divider()

# ==========================================================
# Inspection Report
# ==========================================================

st.subheader("📄 Inspection Report")

report = output["report"]

if isinstance(report, dict):

    col1, col2 = st.columns(2)

    with col1:

        st.write("### General Information")

        st.write(f"**Image:** {report.get('Image','N/A')}")
        st.write(f"**Inspection Time:** {report.get('Inspection Time','N/A')}")
        st.write(f"**Health Score:** {report.get('Health Score','N/A')}%")
        st.write(f"**Health Status:** {report.get('Health Status','N/A')}")

    with col2:

        st.write("### Inspection Summary")

        st.write(f"**Total Defects:** {report.get('Total Defects',0)}")
        st.write("**Detected Defects:**")
        st.json(report.get("Detected Defects", {}))

else:

    st.text(report)

st.divider()

# ==========================================================
# Download Report
# ==========================================================

st.download_button(
    label="📥 Download Inspection Report",
    data=str(report),
    file_name="pcb_inspection_report.txt",
    mime="text/plain"
)

st.divider()

# ==========================================================
# Model Information
# ==========================================================

with st.expander("ℹ Model Information"):

    st.markdown("""
**Model:** YOLOv8 Nano (YOLOv8n)

**Training Epochs:** 30

**Input Size:** 640 × 640

**Dataset:** PCB Defect Dataset

**Number of Classes:** 6

Supported Defects:

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper
""")

# ==========================================================
# Cleanup Temporary File
# ==========================================================

try:

    os.remove(temp_image_path)

except Exception:

    pass

# ==========================================================
# Footer
# ==========================================================

show_footer()