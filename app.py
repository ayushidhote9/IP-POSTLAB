"""
=============================================================================
VISION LAB // STUDIO - Commercial Computer Vision SaaS Dashboard
=============================================================================
Design Language: Dark Mode First (Deep Slate #0E1117, Electric Cyan #38BDF8)
Tech Stack: Python, Streamlit, OpenCV (cv2), NumPy, Pandas, Scikit-learn, Matplotlib, Pillow (PIL)

12 Interactive Modules:
   1. Environment Setup & Library Overview (NEW)
   2. Image Format Conversion & Bitwise Arithmetic
   3. 2D Geometric Transformations
   4. Spatial Enhancement & Segmentation
   5. Spatial Domain Filtering
   6. Image Inpainting (Restoration)
   7. Lossless Compression Analysis
   8. Morphological Operations
   9. Object Detection via Correlation
  10. Top-Hat Transformation
  11. Color Space Transformations
  12. Edge Detection Comparison
=============================================================================
"""

import io
import time
import sys
import platform
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Streamlit Application Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VISION LAB // STUDIO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# High-End Dark Mode SaaS CSS Styling (Deep Slate #0E1117 & Cyan #38BDF8)
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Base Reset - Deep Slate Obsidian Aesthetic */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }

    /* Main Viewport Container */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px;
    }

    /* Top Hero Header Banner */
    .hero-container {
        background: radial-gradient(100% 120% at 50% 0%, rgba(56, 189, 248, 0.14) 0%, rgba(17, 24, 39, 0.7) 60%, rgba(11, 15, 25, 0.95) 100%),
                    linear-gradient(180deg, #111827 0%, #0B0F19 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.75), 0 0 30px -5px rgba(56, 189, 248, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #38BDF8, #818CF8, transparent);
    }
    .hero-badges-row {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
    }
    .badge-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 4px 11px;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .badge-pill-online .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
    }
    .badge-pill-version {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38BDF8;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 11px;
        border-radius: 9999px;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #FFFFFF 20%, #38BDF8 65%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        margin: 0;
        font-weight: 500;
    }

    /* Sidebar Customizations */
    [data-testid="stSidebar"] {
        background-color: #080C14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem !important;
    }
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748B;
        margin: 18px 0 8px 2px;
    }

    /* Module Header Banner */
    .module-card-header {
        background: linear-gradient(180deg, #111827 0%, #0D131F 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 5px solid #38BDF8;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .module-badge {
        font-size: 0.72rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .module-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 3px 0 4px 0;
    }
    .module-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.45;
        margin: 0;
    }

    /* Glassmorphism Visualizer Container Cards */
    .visualizer-card {
        background: #0E1422;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .visualizer-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 10px 30px -5px rgba(56, 189, 248, 0.2);
    }
    .card-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .tag-source {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #CBD5E1;
    }
    .tag-processed {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38BDF8;
    }

    /* Metric & KPI Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #101626 0%, #0C101C 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.45rem !important;
        color: #38BDF8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #94A3B8 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }

    /* High-Contrast Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        padding: 8px 18px !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* High-Contrast Download CTA Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        width: 100%;
        margin-top: 8px;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.5) !important;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        background-color: #0E1422 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #0C101C !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* Form Inputs, Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #101626 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* Keyframe Pulse Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.92); }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Procedural Sample Image Generators (Self-Contained & Instant Startup)
# ---------------------------------------------------------------------------
@st.cache_data
def generate_sample_image(sample_type: str = "Standard CV Test Pattern") -> np.ndarray:
    """Generates synthetic high-resolution test images in RGB uint8 shape (H, W, 3)."""
    h, w = 480, 560
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if sample_type == "Standard CV Test Pattern":
        # Smooth dark cyan-indigo gradient backdrop
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        img[..., 0] = np.uint8((x_coords / w) * 160 + 35)
        img[..., 1] = np.uint8((y_coords / h) * 140 + 45)
        img[..., 2] = np.uint8(((x_coords + y_coords) / (w + h)) * 190 + 35)

        # Reference color bars
        bar_height = 36
        colors = [
            (244, 63, 94),   # Rose Red
            (16, 185, 129),  # Emerald Green
            (59, 130, 246),  # Sky Blue
            (245, 158, 11),  # Amber Yellow
            (6, 182, 212),   # Cyan
            (168, 85, 247),  # Purple
            (248, 250, 252), # Crisp White
            (15, 23, 42)     # Slate Black
        ]
        cw = w // len(colors)
        for i, c in enumerate(colors):
            img[12:12+bar_height, i*cw:(i+1)*cw] = c

        # Shapes for Edge, Filtering, Morphology & Correlation
        cv2.circle(img, (140, 180), 55, (255, 215, 0), -1)
        cv2.circle(img, (140, 180), 30, (14, 116, 144), -1)
        cv2.rectangle(img, (260, 120), (370, 230), (6, 182, 212), -1)
        cv2.rectangle(img, (285, 145), (345, 205), (190, 24, 93), -1)

        # Star / Diamond shape
        pts = np.array([[460, 120], [500, 175], [460, 230], [420, 175]], np.int32)
        cv2.fillPoly(img, [pts], (234, 88, 12))

        # Fine Checkerboard pattern for frequency tests
        grid_y, grid_x = 280, 40
        for r in range(6):
            for c in range(6):
                color = (255, 255, 255) if (r + c) % 2 == 0 else (20, 25, 35)
                cv2.rectangle(img, (grid_x + c*20, grid_y + r*20), (grid_x + (c+1)*20, grid_y + (r+1)*20), color, -1)

        # Concentric circles for frequency & blur
        center = (280, 340)
        for radius in range(12, 65, 12):
            cv2.circle(img, center, radius, (240, 240, 240), 2)

        # Crisp High-Contrast Typography
        cv2.putText(img, "VISION LAB STUDIO", (190, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "NEURAL SUITE 2026", (370, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

    elif sample_type == "Damaged Photo (For Inpainting)":
        for y in range(h):
            r = int(50 + 120 * (y / h))
            g = int(90 + 70 * (y / h))
            b = int(180 - 70 * (y / h))
            img[y, :] = (min(255, r), min(255, g), min(255, b))

        pts_mountain = np.array([[0, 320], [120, 200], [220, 260], [360, 160], [480, 280], [560, 210], [560, 480], [0, 480]], np.int32)
        cv2.fillPoly(img, [pts_mountain], (30, 41, 59))
        cv2.circle(img, (440, 90), 40, (254, 240, 138), -1)

        # Add synthetic damages
        cv2.line(img, (60, 80), (220, 340), (255, 255, 255), 4)
        cv2.line(img, (200, 100), (320, 280), (255, 255, 255), 3)
        cv2.line(img, (400, 150), (430, 410), (255, 255, 255), 5)
        cv2.putText(img, "RESTORE ME", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.circle(img, (280, 160), 16, (255, 255, 255), -1)

    elif sample_type == "Document Scan (Uneven Illumination)":
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist_from_corner = np.sqrt((x_coords - 0)**2 + (y_coords - 0)**2)
        illumination = 245 - 160 * (dist_from_corner / np.sqrt(h**2 + w**2))
        img[..., 0] = np.uint8(np.clip(illumination, 35, 255))
        img[..., 1] = np.uint8(np.clip(illumination, 35, 255))
        img[..., 2] = np.uint8(np.clip(illumination - 12, 25, 255))

        text_lines = [
            "COMMERCIAL VISION INTELLIGENCE PIPELINE",
            "Module 10: Top-Hat & Illumination Flattening",
            "Non-uniform lighting and shadows pose severe",
            "challenges for downstream neural segmenters.",
            "Mathematical morphology extracts high-frequency",
            "micro-structures across non-uniform baselines.",
            "Normalized cross-correlation localizes features.",
            "Lossless containers preserve raw bit precision."
        ]
        for idx, line in enumerate(text_lines):
            y_pos = 70 + idx * 45
            cv2.putText(img, line, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 23, 42), 2, cv2.LINE_AA)

    elif sample_type == "Geometric Coins & Particles":
        img[:] = (20, 27, 40)
        centers = [
            (90, 110), (220, 100), (360, 120), (480, 90),
            (110, 240), (240, 250), (370, 230), (490, 250),
            (100, 380), (210, 390), (340, 370), (470, 390)
        ]
        radii = [35, 28, 42, 30, 26, 44, 32, 28, 38, 30, 34, 40]
        colors = [
            (210, 180, 120), (220, 220, 220), (190, 140, 70), (200, 200, 210),
            (180, 130, 60), (230, 230, 230), (215, 175, 95), (205, 205, 205),
            (225, 185, 105), (195, 145, 75), (210, 210, 220), (220, 180, 90)
        ]
        for c, r, col in zip(centers, radii, colors):
            cv2.circle(img, c, r, col, -1)
            cv2.circle(img, c, r, (255, 255, 255), 2)
            cv2.circle(img, (c[0]-r//4, c[1]-r//4), r//3, (min(255, col[0]+40), min(255, col[1]+40), min(255, col[2]+40)), -1)

    return img


# ---------------------------------------------------------------------------
# Helper Utilities: Image Conversions, Downloads & Dark Matplotlib Styles
# ---------------------------------------------------------------------------
def to_pil_download_bytes(image: np.ndarray, file_format: str = "PNG") -> bytes:
    """Converts a NumPy RGB/Grayscale image to downloadable bytes."""
    buf = io.BytesIO()
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode='RGBA')
    else:
        pil_img = Image.fromarray(image, mode='RGB')
    pil_img.save(buf, format=file_format)
    return buf.getvalue()


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensures image is 3-channel RGB."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Ensures image is single-channel Grayscale uint8."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def render_split_visualizer(orig_img: np.ndarray, proc_img: np.ndarray,
                            proc_caption: str, download_filename: str,
                            orig_caption: str = "Original Source Input"):
    """Renders a sleek side-by-side split screen with SaaS badge headers and download CTA."""
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">⚡ SOURCE INPUT</span>', unsafe_allow_html=True)
        st.image(orig_img, caption=orig_caption, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">✨ PROCESSED OUTPUT</span>', unsafe_allow_html=True)
        st.image(proc_img, caption=proc_caption, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label=f"⬇️ Download Processed Image ({download_filename.split('.')[-1].upper()})",
            data=to_pil_download_bytes(proc_img),
            file_name=download_filename,
            mime="image/png"
        )


def configure_dark_matplotlib():
    """Configures modern dark aesthetic for Matplotlib charts."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#0E1422'
    plt.rcParams['axes.facecolor'] = '#0E1422'
    plt.rcParams['axes.edgecolor'] = '#1E293B'
    plt.rcParams['grid.color'] = '#1E293B'
    plt.rcParams['text.color'] = '#F8FAFC'
    plt.rcParams['xtick.color'] = '#94A3B8'
    plt.rcParams['ytick.color'] = '#94A3B8'


def render_module_header(mod_num: int, title: str, description: str):
    """Renders sleek top banner for active module."""
    st.markdown(f"""
    <div class="module-card-header">
        <div class="module-badge">MODULE // {mod_num:02d}</div>
        <div class="module-title">{title}</div>
        <p class="module-desc">{description}</p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# MODULE 1: Environment Setup & Library Overview (NEW)
# ===========================================================================
def render_module_env_setup(image: np.ndarray):
    render_module_header(
        1,
        "Environment Setup & Library Overview",
        "Setup, configure, and verify the Python digital image processing development environment "
        "including PyCharm IDE, Jupyter3 / JupyterLab, and foundational libraries: OpenCV, NumPy, Pandas, Scikit-learn, and Matplotlib."
    )

    st.markdown("""
    <div style="background: #111827; border: 1.5px solid rgba(56, 189, 248, 0.25); border-left: 5px solid #38BDF8; border-radius: 14px; padding: 18px 22px; margin-bottom: 22px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);">
        <div style="font-size: 0.72rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.1em; text-transform: uppercase;">LABORATORY EXPERIMENT AIM</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0 6px 0;">Introduction to Python & Setup of Development Environment for Image Processing</div>
        <p style="font-size: 0.92rem; color: #94A3B8; margin: 0; line-height: 1.5;">
            <strong>Formal Aim:</strong> Introduction to Python and Setup of Development Environment (necessary tools and libraries) for Image Processing. 
            (Including IDE PyCharm, Jupyter3/JupyterLab, Installation of library OpenCV, NumPy, pandas, scikit-learn, matplotlib for Image processing activity.)
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_ide, tab_libs, tab_sandbox, tab_viva = st.tabs([
        "🖥️ IDE Setup (PyCharm & Jupyter3)",
        "📦 Live System Check & Diagnostics",
        "🧪 Interactive 5-Library Sandbox",
        "📝 Lab Viva & Theory Checkpoints"
    ])

    with tab_ide:
        st.markdown("### 🖥️ Integrated Development Environments (IDEs)")
        st.markdown(
            "Computer vision pipelines require environments supporting high-dimensional array inspection, "
            "visual debugging, and modular scripting. Below is the setup guide for PyCharm and Jupyter3."
        )

        col_pycharm, col_jupyter = st.columns(2)

        with col_pycharm:
            st.markdown("""
            <div class="visualizer-card">
                <span class="card-tag tag-processed">JETBRAINS PYCHARM</span>
                <h4 style="margin: 4px 0 10px 0; color: #F8FAFC;">PyCharm IDE Configuration</h4>
                <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45;">
                    PyCharm is the industry-standard IDE for professional Python engineering with built-in scientific tools and array inspection.
                </p>
                <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
                    <strong>Step 1: Download & Install</strong><br>
                    Download PyCharm Community Edition (Free) or Professional from <a href="https://www.jetbrains.com/pycharm/" target="_blank" style="color: #38BDF8;">jetbrains.com</a>.<br><br>
                    <strong>Step 2: Interpreter Configuration</strong><br>
                    Go to <code>File &gt; Settings &gt; Project &gt; Python Interpreter</code>. Click <em>Add Interpreter</em> and configure a Python 3.10+ Virtualenv.<br><br>
                    <strong>Step 3: Dependency Installation</strong><br>
                    Open PyCharm's built-in Terminal (<code>Alt + F12</code>) and execute:<br>
                    <code>pip install opencv-python numpy pandas scikit-learn matplotlib pillow streamlit</code><br><br>
                    <strong>Step 4: Interactive Array Debugging</strong><br>
                    Set a breakpoint (click gutter next to line number). Run Debugger (<code>Shift + F9</code>). Inspect NumPy image matrices directly in the <strong>Variables</strong> window or SciView.<br><br>
                    <strong>Key Shortcuts:</strong><br>
                    • Run: <code>Shift + F10</code> &nbsp; • Debug: <code>Shift + F9</code><br>
                    • Project View: <code>Alt + 1</code> &nbsp; • Settings: <code>Ctrl + Alt + S</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_jupyter:
            st.markdown("""
            <div class="visualizer-card">
                <span class="card-tag tag-source">JUPYTER3 / JUPYTERLAB</span>
                <h4 style="margin: 4px 0 10px 0; color: #F8FAFC;">Jupyter3 Notebook Setup</h4>
                <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45;">
                    Jupyter3 provides a web-based, cell-by-cell interactive notebook interface ideal for exploratory computer vision and algorithm prototyping.
                </p>
                <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
                    <strong>Step 1: Installation</strong><br>
                    Install Jupyter Notebook / JupyterLab via pip:<br>
                    <code>pip install notebook jupyterlab</code><br><br>
                    <strong>Step 2: Launch Notebook Server</strong><br>
                    Launch the browser workspace from your project folder:<br>
                    <code>jupyter notebook</code> or <code>jupyter lab</code><br><br>
                    <strong>Step 3: Inline Image Rendering Magic</strong><br>
                    In your first notebook cell, enable inline Matplotlib graphics:<br>
                    <code>%matplotlib inline</code><br>
                    <code>import cv2, numpy as np, matplotlib.pyplot as plt</code><br><br>
                    <strong>Step 4: Cell-by-Cell Visualization</strong><br>
                    Load an image with <code>cv2.imread()</code>, convert BGR to RGB, and display instantly with <code>plt.imshow(img)</code> without freezing windows.<br><br>
                    <strong>Key Shortcuts:</strong><br>
                    • Run Cell: <code>Shift + Enter</code> &nbsp; • Insert Below: <code>Esc + B</code><br>
                    • Markdown: <code>Esc + M</code> &nbsp; • Code Mode: <code>Esc + Y</code> &nbsp; • Delete: <code>Esc + D + D</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ IDE Comparison: PyCharm vs Jupyter3")
        comparison_data = {
            "Dimension": ["Execution Paradigm", "Best Suited For", "Debugging Capabilities", "Image Inspection", "Version Control (Git)", "Performance"],
            "PyCharm IDE": [
                "Script & Project execution",
                "Production vision pipelines, complex modules & apps",
                "Full interactive visual debugger, breakpoints & watches",
                "SciView data matrix inspector, array shape visualizer",
                "Seamless built-in Git GUI with visual diffs & history",
                "High performance compiled execution"
            ],
            "Jupyter3 Notebook": [
                "Interactive Cell-by-Cell evaluation",
                "Exploratory research, quick filters & academic reporting",
                "Inline print statements, ipdb step-through",
                "Instant inline %matplotlib plots below cells",
                "JSON-based diffs (nbdime recommended)",
                "Interactive memory-resident state"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    with tab_libs:
        st.markdown("### 📦 Live System Check & Dependency Diagnostics")
        st.markdown("Real-time verification of installed libraries in your active Python interpreter environment.")

        py_ver = sys.version.split()[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Python Runtime", f"v{py_ver}", "64-bit CPython")
        with c2:
            st.metric("OpenCV (cv2)", f"v{cv2.__version__}", "Image Processing Core")
        with c3:
            st.metric("NumPy", f"v{np.__version__}", "N-D Tensor Engine")
        with c4:
            st.metric("Pandas", f"v{pd.__version__}", "Pixel Tabular Engine")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Scikit-learn", f"v{sklearn.__version__}", "ML & Clustering")
        with c6:
            st.metric("Matplotlib", f"v{plt.matplotlib.__version__}", "Plotting & Histograms")
        with c7:
            st.metric("Pillow (PIL)", f"v{Image.__version__}", "Raster Image I/O")
        with c8:
            st.metric("Streamlit", f"v{st.__version__}", "Web UI Engine")

        st.markdown("#### 📋 Core Library Roles in Computer Vision")
        lib_roles = [
            ("OpenCV (cv2)", cv2.__version__, "Real-time computer vision, filtering, transformations, morphology, feature detection, inpainting, and video processing."),
            ("NumPy (numpy)", np.__version__, "High-performance N-dimensional array processing. Images are treated as uint8 matrices of shape (H, W, C)."),
            ("Pandas (pandas)", pd.__version__, "Tabular structuring of image metadata, pixel intensities, bounding-box annotations (YOLO/COCO), and dataset indexing."),
            ("Scikit-learn (sklearn)", sklearn.__version__, "Machine learning algorithms: K-Means color quantization, PCA dimensionality reduction, image feature normalization, and classification."),
            ("Matplotlib (matplotlib)", plt.matplotlib.__version__, "Scientific visualization: Pixel intensity histograms, 2D/3D surface plots, false-color heatmap palettes, and before/after comparisons.")
        ]
        role_df = pd.DataFrame(lib_roles, columns=["Library", "Detected Version", "Primary Role in Vision Pipeline"])
        st.dataframe(role_df, use_container_width=True, hide_index=True)

        st.markdown("#### 💻 Unified Installation Terminal Command")
        st.code("pip install opencv-python numpy pandas scikit-learn matplotlib pillow streamlit", language="bash")

        req_content = "streamlit>=1.30.0\nopencv-python>=4.8.0\nnumpy>=1.24.0\npandas>=2.0.0\nscikit-learn>=1.3.0\nmatplotlib>=3.8.0\npillow>=10.0.0\n"
        st.download_button(
            label="⬇️ Download requirements.txt File",
            data=req_content,
            file_name="requirements.txt",
            mime="text/plain"
        )

    with tab_sandbox:
        st.markdown("### 🧪 Interactive 5-Library Sandbox & Quick-Test Runner")
        st.markdown("Execute live tests for each of the 5 foundational libraries using the active image buffer.")

        sandbox_lib = st.radio(
            "Select Library Engine to Test:",
            ["1. NumPy (Array Math & Slicing)", "2. OpenCV (Color Conversion & Annotation)", "3. Pandas (Pixel Tabular Structuring)", "4. Scikit-learn (K-Means Color Quantization)", "5. Matplotlib (Intensity & CDF Plot)"],
            horizontal=True
        )

        if "NumPy" in sandbox_lib:
            st.markdown("##### 🔢 NumPy Array Operations")
            c_np1, c_np2 = st.columns([1, 1.5])
            with c_np1:
                st.info(f"**Array Properties:**\n- Shape: `{image.shape}`\n- Data Type: `{image.dtype}`\n- Total Elements: `{image.size:,}` pixels\n- Memory Footprint: `{image.nbytes / 1024:.1f} KB`\n- Value Range: `[{np.min(image)}, {np.max(image)}]`\n- Mean ± Std: `{np.mean(image):.1f} ± {np.std(image):.1f}`")
                np_op = st.selectbox("NumPy Manipulation:", ["Channel Inversion (255 - I)", "Red Channel Isolation", "Green Channel Isolation", "Blue Channel Isolation", "Threshold Mask Array"])
                if np_op == "Channel Inversion (255 - I)":
                    np_result = 255 - image
                elif np_op == "Red Channel Isolation":
                    np_result = image.copy()
                    np_result[:, :, 1:] = 0
                elif np_op == "Green Channel Isolation":
                    np_result = image.copy()
                    np_result[:, :, 0] = 0
                    np_result[:, :, 2] = 0
                elif np_op == "Blue Channel Isolation":
                    np_result = image.copy()
                    np_result[:, :, 0:2] = 0
                else:
                    np_result = np.where(image > 128, 255, 0).astype(np.uint8)
            with c_np2:
                render_split_visualizer(image, np_result, f"NumPy Result: {np_op}", "module1_numpy_test.png")

        elif "OpenCV" in sandbox_lib:
            st.markdown("##### 📷 OpenCV Vision Processing")
            c_cv1, c_cv2 = st.columns([1, 1.5])
            with c_cv1:
                cv_demo = st.selectbox("OpenCV Test Operation:", ["Color Conversion (RGB to Grayscale)", "Gaussian Blur Smoothing", "Geometric Primitives & Annotation Overlay"])
                if cv_demo == "Color Conversion (RGB to Grayscale)":
                    cv_out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif cv_demo == "Gaussian Blur Smoothing":
                    blur_k = st.slider("Gaussian Kernel Size (Odd)", 3, 31, 9, step=2)
                    cv_out = cv2.GaussianBlur(image, (blur_k, blur_k), 0)
                else:
                    cv_out = image.copy()
                    h_c, w_c = cv_out.shape[:2]
                    cv2.rectangle(cv_out, (w_c//6, h_c//6), (5*w_c//6, 5*h_c//6), (16, 185, 129), 3)
                    cv2.circle(cv_out, (w_c//2, h_c//2), min(h_c, w_c)//4, (56, 189, 248), 3)
                    cv2.putText(cv_out, "OPENCV READY", (w_c//4, h_c//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            with c_cv2:
                render_split_visualizer(image, cv_out, f"OpenCV Result: {cv_demo}", "module1_opencv_test.png")

        elif "Pandas" in sandbox_lib:
            st.markdown("##### 🐼 Pandas Tabular Pixel Structuring")
            st.markdown("Sample an image region of interest (ROI) and convert spatial pixel matrices into structured tabular `DataFrame` format.")
            patch_dim = st.slider("ROI Sample Grid Dimension (N × N pixels):", 8, 32, 16, step=4)
            patch = image[:patch_dim, :patch_dim]

            records = []
            for y in range(patch_dim):
                for x in range(patch_dim):
                    r, g, b = patch[y, x][:3]
                    luma = int(0.299 * r + 0.587 * g + 0.114 * b)
                    records.append({"Pixel_Index": y * patch_dim + x, "Coord_X": x, "Coord_Y": y, "Red": int(r), "Green": int(g), "Blue": int(b), "Luminance": luma})
            df_pixels = pd.DataFrame(records)

            c_df1, c_df2 = st.columns([1.5, 1])
            with c_df1:
                st.markdown(f"**Sampled DataFrame (`{len(df_pixels)}` pixel rows × 7 attributes):**")
                st.dataframe(df_pixels.head(10), use_container_width=True)
                csv_bytes = df_pixels.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Export Pixel Dataset (CSV)", csv_bytes, "pixel_dataset.csv", "text/csv")
            with c_df2:
                st.markdown("**Descriptive Statistics (`df.describe()`):**")
                st.dataframe(df_pixels[['Red', 'Green', 'Blue', 'Luminance']].describe().round(1), use_container_width=True)

        elif "Scikit-learn" in sandbox_lib:
            st.markdown("##### 🤖 Scikit-learn K-Means Color Quantization")
            st.markdown("Train an unsupervised **K-Means clustering algorithm** on pixel color vectors $(R, G, B)$ to segment the image into $K$ dominant palette clusters.")
            n_clusters = st.slider("Number of Color Clusters (K):", 2, 8, 4)

            with st.spinner(f"Fitting scikit-learn KMeans with K={n_clusters} clusters..."):
                h_k, w_k = image.shape[:2]
                pixels = image.reshape(-1, 3).astype(np.float32)
                sample_indices = np.random.choice(pixels.shape[0], min(5000, pixels.shape[0]), replace=False)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(pixels[sample_indices])
                labels = kmeans.predict(pixels)
                quantized_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
                quantized_img = quantized_pixels.reshape((h_k, w_k, 3))

            render_split_visualizer(image, quantized_img, f"Scikit-learn Quantized (K={n_clusters} clusters)", "module1_sklearn_kmeans.png")

            st.markdown(f"**Extracted {n_clusters} Dominant Color Centroids (RGB Swatches):**")
            swatch_cols = st.columns(n_clusters)
            for idx, center in enumerate(kmeans.cluster_centers_):
                r, g, b = [int(v) for v in center[:3]]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                with swatch_cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: {hex_color}; height: 45px; border-radius: 8px; border: 1.5px solid #38BDF8; margin-bottom: 4px;"></div>
                    <div style="font-size: 0.76rem; font-weight: 700; text-align: center; color: #F8FAFC;">Cluster {idx+1}<br><code>{hex_color}</code></div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("##### 📈 Matplotlib Multi-Channel Distribution")
            configure_dark_matplotlib()
            fig, ax = plt.subplots(figsize=(8.5, 3))
            colors = ('#F43F5E', '#10B981', '#38BDF8')
            labels = ('Red Channel', 'Green Channel', 'Blue Channel')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=col, label=labels[i], linewidth=1.8, alpha=0.85)
            ax.set_title("Matplotlib Pixel Intensity Distribution", fontsize=11, fontweight='600')
            ax.set_xlabel("Pixel Value (0 - 255)", fontsize=9)
            ax.set_ylabel("Pixel Count", fontsize=9)
            ax.set_xlim([0, 256])
            ax.legend(loc="upper right")
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_viva:
        st.markdown("### 📝 Laboratory Viva & Conceptual Checkpoints")
        viva_qa = [
            ("Q1: Why is Python preferred over C++ or Java for digital image processing?",
             "Python combines clean, expressive syntax with high-performance C/C++ native acceleration through NumPy and OpenCV bindings. It allows developers and researchers to rapidly prototype complex algorithms in a few lines of code while executing numerical heavy-lifting at compiled native speeds."),
            ("Q2: Why does OpenCV represent images as NumPy ndarray objects?",
             "By standardizing on NumPy ndarray, OpenCV gains seamless zero-copy interoperability with the entire Python scientific ecosystem (pandas, scikit-learn, SciPy, PyTorch, Matplotlib). An image is naturally modeled as an array of shape (Height, Width, Channels) with uint8 precision [0, 255]."),
            ("Q3: Why does OpenCV read color images in BGR order rather than RGB?",
             "Historical convention: When OpenCV was originally developed by Intel in 1999, BGR was the standard memory storage format used by camera manufacturers and Windows bitmap (DIB) drivers. When displaying OpenCV images in Matplotlib or PIL, BGR must be converted to RGB using cv2.cvtColor(img, cv2.COLOR_BGR2RGB)."),
            ("Q4: What is the role of pandas in computer vision pipelines?",
             "While OpenCV processes raw pixel rasters, pandas handles structured metadata: bounding box annotations (YOLO/COCO formats with x, y, w, h coordinates), dataset ground-truth labels, cross-validation splits, and pixel statistical summaries."),
            ("Q5: How does scikit-learn complement OpenCV in image analysis?",
             "OpenCV provides low-level spatial filtering and transformation primitives, while scikit-learn provides high-level machine learning: unsupervised clustering (K-Means color segmentation), feature normalization (StandardScaler), dimensionality reduction (PCA for eigenfaces), and classification (SVMs, Random Forests)."),
            ("Q6: When should an engineer choose PyCharm over Jupyter Notebook?",
             "Choose Jupyter3 for exploratory data analysis, visual filter experimentation, and step-by-step reporting. Choose PyCharm for building modular production code, multi-file software architectures, Streamlit apps, packaging libraries, and advanced multi-threaded debugging.")
        ]
        for q, a in viva_qa:
            with st.expander(f"❓ {q}", expanded=False):
                st.markdown(a)


# ===========================================================================
# MODULE 2: Image Format Conversion & Bitwise Arithmetic
# ===========================================================================
def render_module_format_bitwise(image: np.ndarray):
    render_module_header(
        2,
        "Image Format Conversion & Bitwise Arithmetic",
        "Convert color spaces into grayscale or segmented binary bitplanes, and execute Boolean logic "
        "and saturated image arithmetic using synthesized spatial masks or scalar modulation."
    )

    tab1, tab2 = st.tabs(["🔲 Color & Binary Conversions", "⚡ Bitwise & Saturated Arithmetic"])

    with tab1:
        c_ctrl1, c_ctrl2 = st.columns(2)
        with c_ctrl1:
            conv_mode = st.radio(
                "Target Bitplane / Format:",
                ["Grayscale (Luminance)", "Binary (Global Manual Threshold)", "Binary (Otsu's Adaptive Optimal)", "Inverted Binary"],
                index=0
            )
        with c_ctrl2:
            thresh_val = st.slider("Manual Threshold Cutoff (0-255)", min_value=0, max_value=255, value=128,
                                   disabled=("Otsu" in conv_mode or "Grayscale" in conv_mode))

        with st.spinner("Processing bitplane conversion..."):
            gray = ensure_gray(image)
            if conv_mode == "Grayscale (Luminance)":
                result = gray
                formula = r"I_{\text{gray}} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B"
            elif conv_mode == "Binary (Global Manual Threshold)":
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                formula = rf"I_{{bin}}(x,y) = 255 \text{{ if }} I(x,y) \ge {thresh_val} \text{{ else }} 0"
            elif conv_mode == "Binary (Otsu's Adaptive Optimal)":
                otsu_val, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                formula = rf"\text{{Optimal Otsu Threshold Calculated: }} T^* = {int(otsu_val)}"
            else:
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                formula = rf"I_{{inv}}(x,y) = 0 \text{{ if }} I(x,y) \ge {thresh_val} \text{{ else }} 255"

        st.latex(formula)
        render_split_visualizer(image, result, f"Result: {conv_mode}", "module2_format_converted.png")
        st.toast("Format conversion completed successfully!", icon="✨")

    with tab2:
        c_op, c_mode = st.columns(2)
        with c_op:
            operation = st.selectbox(
                "Boolean / Arithmetic Operation:",
                [
                    "Bitwise AND (I ∧ M)",
                    "Bitwise OR (I ∨ M)",
                    "Bitwise XOR (I ⊕ M)",
                    "Bitwise NOT (¬I)",
                    "Saturated Addition (I ⊞ M)",
                    "Saturated Subtraction (I ⊟ M)",
                    "Alpha Linear Blending (α·I + (1-α)·M)"
                ]
            )
        with c_mode:
            operand_type = st.selectbox(
                "Secondary Operand / Mask Generator:",
                ["Generated Spatial Mask", "Uniform Scalar Value Slider"]
            )

        h, w = image.shape[:2]

        if operand_type == "Generated Spatial Mask":
            mask_shape = st.selectbox("Aperture Geometry:", ["Circular Aperture", "Centered Rectangle", "Checkerboard Grid", "Horizontal Gradient"])
            mask = np.zeros((h, w), dtype=np.uint8)
            if mask_shape == "Circular Aperture":
                cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)
            elif mask_shape == "Centered Rectangle":
                cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 255, -1)
            elif mask_shape == "Checkerboard Grid":
                step = 40
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        if ((x // step) + (y // step)) % 2 == 0:
                            cv2.rectangle(mask, (x, y), (min(x + step, w), min(y + step, h)), 255, -1)
            elif mask_shape == "Horizontal Gradient":
                for x in range(w):
                    mask[:, x] = int((x / w) * 255)
            operand_display = mask
            operand_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            scalar_val = st.slider("Modulation Scalar Intensity (0 - 255):", min_value=0, max_value=255, value=100)
            mask = np.full((h, w), scalar_val, dtype=np.uint8)
            operand_display = mask
            operand_3ch = np.full((h, w, 3), scalar_val, dtype=np.uint8)

        with st.spinner("Computing Boolean matrix transformation..."):
            if "Bitwise AND" in operation:
                result = cv2.bitwise_and(image, operand_3ch)
            elif "Bitwise OR" in operation:
                result = cv2.bitwise_or(image, operand_3ch)
            elif "Bitwise XOR" in operation:
                result = cv2.bitwise_xor(image, operand_3ch)
            elif "Bitwise NOT" in operation:
                result = cv2.bitwise_not(image)
            elif "Saturated Addition" in operation:
                result = cv2.add(image, operand_3ch)
            elif "Saturated Subtraction" in operation:
                result = cv2.subtract(image, operand_3ch)
            else:
                alpha = st.slider("Alpha Blend Weight (α)", 0.0, 1.0, 0.6, step=0.05)
                result = cv2.addWeighted(image, alpha, operand_3ch, 1.0 - alpha, 0.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">INPUT (I)</span>', unsafe_allow_html=True)
            st.image(image, caption="Primary Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">OPERAND (M)</span>', unsafe_allow_html=True)
            st.image(operand_display, caption="Spatial Mask / Scalar", use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">OUTPUT RESULT</span>', unsafe_allow_html=True)
            st.image(result, caption=f"Result: {operation.split('(')[0]}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Processed Image",
                data=to_pil_download_bytes(result),
                file_name="module2_bitwise_output.png",
                mime="image/png"
            )


# ===========================================================================
# MODULE 3: 2D Geometric Transformations
# ===========================================================================
def render_module_geometric(image: np.ndarray):
    render_module_header(
        3,
        "2D Geometric Transformations",
        "Apply spatial coordinate mappings via affine transformation matrices, high-order interpolations, "
        "and Euclidean coordinate transformations with boundary extrapolation."
    )

    h, w = image.shape[:2]
    col_ctrl, col_prev = st.columns([1, 1.6])

    with col_ctrl:
        st.markdown("#### 🎛️ Transformation Controls")
        trans_type = st.selectbox(
            "Transformation Mode:",
            ["Translation", "Rotation & Zoom", "Scaling (Interpolation)", "Shearing (Affine)", "Reflection (Flip)", "ROI Cropping"]
        )

        border_choice = st.selectbox(
            "Border Extrapolation Mode:",
            ["BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP"]
        )
        border_modes = {
            "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
            "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
            "BORDER_REFLECT": cv2.BORDER_REFLECT,
            "BORDER_WRAP": cv2.BORDER_WRAP
        }
        b_mode = border_modes[border_choice]

        if trans_type == "Translation":
            tx = st.slider("Translation ΔX (px)", -w // 2, w // 2, 50)
            ty = st.slider("Translation ΔY (px)", -h // 2, h // 2, 35)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            result = cv2.warpAffine(image, M, (w, h), borderMode=b_mode)
            math_expr = rf"M = \begin{{bmatrix}} 1 & 0 & {tx} \\ 0 & 1 & {ty} \end{{bmatrix}}"

        elif trans_type == "Rotation & Zoom":
            angle = st.slider("Rotation Angle (θ Degrees)", 0.0, 360.0, 45.0, step=1.0)
            scale = st.slider("Zoom Scale Multiplier", 0.2, 2.5, 1.0, step=0.05)
            center_x = st.slider("Anchor Center X", 0, w, w // 2)
            center_y = st.slider("Anchor Center Y", 0, h, h // 2)
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
            result = cv2.warpAffine(image, M, (w, h), borderMode=b_mode)
            math_expr = rf"\text{{Rotated by }} {angle}^\circ \text{{ around }} ({center_x}, {center_y}) \text{{ at }} {scale}\times"

        elif trans_type == "Scaling (Interpolation)":
            scale_factor = st.slider("Resolution Scale Multiplier", 0.1, 3.0, 1.4, step=0.1)
            interp_method = st.selectbox(
                "Resampling Algorithm:",
                ["INTER_LINEAR (Bilinear)", "INTER_NEAREST (Nearest Neighbor)", "INTER_CUBIC (Bicubic)", "INTER_LANCZOS4 (Lanczos-8)"]
            )
            interp_dict = {
                "INTER_LINEAR (Bilinear)": cv2.INTER_LINEAR,
                "INTER_NEAREST (Nearest Neighbor)": cv2.INTER_NEAREST,
                "INTER_CUBIC (Bicubic)": cv2.INTER_CUBIC,
                "INTER_LANCZOS4 (Lanczos-8)": cv2.INTER_LANCZOS4
            }
            new_w = max(1, int(w * scale_factor))
            new_h = max(1, int(h * scale_factor))
            result = cv2.resize(image, (new_w, new_h), interpolation=interp_dict[interp_method])
            math_expr = rf"\text{{Resized: }} {w}\times{h} \to {new_w}\times{new_h} \text{{ using }} {interp_method.split()[0]}"

        elif trans_type == "Shearing (Affine)":
            shx = st.slider("Shear X (sh_x)", -1.0, 1.0, 0.25, step=0.05)
            shy = st.slider("Shear Y (sh_y)", -1.0, 1.0, 0.0, step=0.05)
            M = np.float32([[1, shx, 0], [shy, 1, 0]])
            result = cv2.warpAffine(image, M, (int(w * (1 + abs(shx))), int(h * (1 + abs(shy)))), borderMode=b_mode)
            math_expr = rf"M_{{shear}} = \begin{{bmatrix}} 1 & {shx:.2f} & 0 \\ {shy:.2f} & 1 & 0 \end{{bmatrix}}"

        elif trans_type == "Reflection (Flip)":
            flip_mode = st.radio("Reflection Symmetry:", ["Horizontal Flip (Y-Axis)", "Vertical Flip (X-Axis)", "Both Axes (Origin)"])
            flip_code = 1 if "Horizontal" in flip_mode else (0 if "Vertical" in flip_mode else -1)
            result = cv2.flip(image, flip_code)
            math_expr = rf"\text{{Mirrored via }} \texttt{{cv2.flip(I, {flip_code})}}"

        else:
            c_x1, c_x2 = st.slider("Horizontal Bounding Range [Xmin, Xmax]", 0, w, (w // 6, 5 * w // 6))
            c_y1, c_y2 = st.slider("Vertical Bounding Range [Ymin, Ymax]", 0, h, (h // 6, 5 * h // 6))
            if c_x2 > c_x1 and c_y2 > c_y1:
                result = image[c_y1:c_y2, c_x1:c_x2].copy()
                math_expr = rf"\text{{Cropped ROI: }} X \in [{c_x1}, {c_x2}], Y \in [{c_y1}, {c_y2}]"
            else:
                result = image.copy()
                math_expr = r"\text{Invalid bounds; displaying original}"

        st.latex(math_expr)

    with col_prev:
        render_split_visualizer(image, result, f"Transformed Result ({result.shape[1]}×{result.shape[0]} px)", "module3_geometric.png")


# ===========================================================================
# MODULE 4: Spatial Enhancement & Segmentation
# ===========================================================================
def render_module_spatial_enhancement(image: np.ndarray):
    render_module_header(
        4,
        "Spatial Enhancement & Segmentation",
        "Enhance dynamic dynamic range via histogram equalization and CLAHE, sharpen subtle boundaries "
        "with Laplacian edge operators, and isolate foreground regions with Otsu and adaptive thresholding."
    )

    tab_hist, tab_sharp, tab_thresh = st.tabs([
        "📊 Histogram Equalization & Distribution",
        "🔪 Spatial Sharpening (Laplacian)",
        "🎯 Adaptive Thresholding & Segmentation"
    ])

    with tab_hist:
        eq_method = st.radio("Equalization Mode:", ["Grayscale Equalization", "Color Equalization (LAB L-Channel CLAHE)"], horizontal=True)

        gray = ensure_gray(image)
        if "Grayscale" in eq_method:
            orig_view = gray
            eq_result = cv2.equalizeHist(gray)
            is_color = False
        else:
            orig_view = image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            eq_lab = cv2.merge((cl, a, b))
            eq_result = cv2.cvtColor(eq_lab, cv2.COLOR_LAB2RGB)
            is_color = True

        render_split_visualizer(orig_view, eq_result, f"Equalized Output ({eq_method.split()[0]})", "module4_equalized.png")

        configure_dark_matplotlib()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

        if not is_color:
            ax1.hist(orig_view.ravel(), 256, [0, 256], color='#38BDF8', alpha=0.85)
            ax1.set_title("Input Intensity Histogram", fontsize=11, fontweight='600')
            ax1.set_xlim([0, 256])
            ax1.grid(True, linestyle='--', alpha=0.3)

            ax2.hist(eq_result.ravel(), 256, [0, 256], color='#10B981', alpha=0.85)
            ax2.set_title("Equalized Intensity Histogram (Uniformized CDF)", fontsize=11, fontweight='600')
            ax2.set_xlim([0, 256])
            ax2.grid(True, linestyle='--', alpha=0.3)
        else:
            colors = ('#F43F5E', '#10B981', '#38BDF8')
            for i, col in enumerate(colors):
                hist_orig = cv2.calcHist([orig_view], [i], None, [256], [0, 256])
                hist_eq = cv2.calcHist([eq_result], [i], None, [256], [0, 256])
                ax1.plot(hist_orig, color=col, alpha=0.85, linewidth=1.5)
                ax2.plot(hist_eq, color=col, alpha=0.85, linewidth=1.5)
            ax1.set_title("Input RGB Histograms", fontsize=11, fontweight='600')
            ax1.set_xlim([0, 256])
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax2.set_title("Equalized RGB Histograms (CLAHE)", fontsize=11, fontweight='600')
            ax2.set_xlim([0, 256])
            ax2.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_sharp:
        c1, c2 = st.columns(2)
        with c1:
            sharp_strength = st.slider("Laplacian Sharpening Gain (α):", 0.1, 3.0, 1.0, step=0.1)
        with c2:
            kernel_choice = st.selectbox("Laplacian Operator Topology:", ["Standard 4-Neighbor", "Diagonal 8-Neighbor"])

        if kernel_choice == "Standard 4-Neighbor":
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        else:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

        img_float = image.astype(np.float32)
        laplacian = cv2.filter2D(img_float, cv2.CV_32F, kernel)
        sharpened = np.clip(img_float + sharp_strength * laplacian, 0, 255).astype(np.uint8)
        laplacian_vis = np.clip(np.abs(laplacian) * 2.5, 0, 255).astype(np.uint8)

        st.latex(r"I_{\text{sharpened}}(x,y) = \text{clip}\left(I(x,y) + \alpha \cdot \nabla^2 I(x,y), 0, 255\right)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">INPUT</span>', unsafe_allow_html=True)
            st.image(image, caption="Original Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">HIGH PASS DETAILS</span>', unsafe_allow_html=True)
            st.image(laplacian_vis, caption="Isolated Laplacian Edges", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">SHARPENED RESULT</span>', unsafe_allow_html=True)
            st.image(sharpened, caption=f"Sharpened Image (α = {sharp_strength})", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Processed Image",
                data=to_pil_download_bytes(sharpened),
                file_name="module4_sharpened.png",
                mime="image/png"
            )

    with tab_thresh:
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            method = st.selectbox(
                "Segmentation Algorithm:",
                [
                    "Manual Global Thresholding",
                    "Otsu's Optimal Bimodal Thresholding",
                    "Adaptive Mean Thresholding",
                    "Adaptive Gaussian Thresholding"
                ]
            )
        with c_opt2:
            gray_src = ensure_gray(image)
            if "Manual" in method:
                thresh_val = st.slider("Global Threshold Cutoff", 0, 255, 128)
                _, thresh_img = cv2.threshold(gray_src, thresh_val, 255, cv2.THRESH_BINARY)
                info_text = f"Manual Cutoff: {thresh_val}"
            elif "Otsu" in method:
                computed_thresh, thresh_img = cv2.threshold(gray_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                info_text = f"Otsu Optimal Threshold Calculated: {int(computed_thresh)}"
            else:
                block_size = st.slider("Adaptive Window Block Size (Odd)", 3, 99, 15, step=2)
                c_const = st.slider("Constant C Subtraction", -30, 30, 4)
                adapt_type = cv2.ADAPTIVE_THRESH_MEAN_C if "Mean" in method else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                thresh_img = cv2.adaptiveThreshold(gray_src, 255, adapt_type, cv2.THRESH_BINARY, block_size, c_const)
                info_text = f"Block Size: {block_size}×{block_size}, C: {c_const}"

        st.info(f"💡 Active Parameter Status: **{info_text}**")
        render_split_visualizer(gray_src, thresh_img, f"Binary Mask: {method}", "module4_threshold.png")


# ===========================================================================
# MODULE 5: Spatial Domain Filtering
# ===========================================================================
def render_module_spatial_filtering(image: np.ndarray):
    render_module_header(
        5,
        "Spatial Domain Filtering",
        "Benchmark linear smoothing (Averaging Box Blur, Gaussian Filter) against non-linear operators "
        "(Median Filter for impulsive noise removal, Bilateral Filter for edge-preserving denoising)."
    )

    c_noise, c_k = st.columns(2)
    with c_noise:
        noise_type = st.selectbox("Simulate Sensor Noise Injection:", ["None (Clean Input)", "Salt & Pepper Noise", "Gaussian Noise"])
    with c_k:
        ksize = st.slider("Filter Kernel Dimension (Odd Window)", min_value=3, max_value=31, value=5, step=2)

    noisy_image = image.copy()
    if noise_type == "Salt & Pepper Noise":
        p_noise = 0.05
        mask_rnd = np.random.rand(*noisy_image.shape[:2])
        noisy_image[mask_rnd < (p_noise / 2)] = 0
        noisy_image[mask_rnd > 1 - (p_noise / 2)] = 255
    elif noise_type == "Gaussian Noise":
        gauss = np.random.normal(0, 20, noisy_image.shape).astype(np.float32)
        noisy_image = np.clip(noisy_image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    view_mode = st.radio("Display View:", ["Interactive Single Filter Deep-Dive", "All 4 Spatial Filters Comparison Grid"], horizontal=True)

    if view_mode == "Interactive Single Filter Deep-Dive":
        c_filter, c_param = st.columns(2)
        with c_filter:
            chosen_filter = st.selectbox(
                "Spatial Filter Algorithm:",
                ["Averaging Filter (Box Blur)", "Gaussian Filter", "Median Filter", "Bilateral Filter"]
            )

        with c_param:
            if chosen_filter == "Gaussian Filter":
                sigma = st.slider("Gaussian Sigma (0 = auto compute from ksize):", 0.0, 15.0, 0.0, step=0.5)
                with st.spinner("Applying Gaussian convolution..."):
                    filtered = cv2.GaussianBlur(noisy_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            elif chosen_filter == "Median Filter":
                st.caption(f"Median rank filtering across {ksize}×{ksize} pixel neighborhood.")
                with st.spinner("Computing rank order statistics..."):
                    filtered = cv2.medianBlur(noisy_image, ksize)
            elif chosen_filter == "Bilateral Filter":
                sigma_color = st.slider("Bilateral Sigma Color:", 10.0, 200.0, 75.0, step=5.0)
                sigma_space = st.slider("Bilateral Sigma Space:", 10.0, 200.0, 75.0, step=5.0)
                with st.spinner("Executing non-linear bilateral range & spatial filtering..."):
                    filtered = cv2.bilateralFilter(noisy_image, d=ksize, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            else:
                st.caption(f"Uniform averaging box blur kernel of dimensions {ksize}×{ksize}.")
                with st.spinner("Computing box convolution..."):
                    filtered = cv2.blur(noisy_image, (ksize, ksize))

        render_split_visualizer(noisy_image, filtered, f"Filtered Output: {chosen_filter} (k={ksize})", f"module5_{chosen_filter.split()[0].lower()}.png")

    else:
        with st.spinner("Rendering 4-quadrant comparative filter benchmark..."):
            f_box = cv2.blur(noisy_image, (ksize, ksize))
            f_gauss = cv2.GaussianBlur(noisy_image, (ksize, ksize), 0)
            f_median = cv2.medianBlur(noisy_image, ksize)
            f_bilateral = cv2.bilateralFilter(noisy_image, d=ksize, sigmaColor=75, sigmaSpace=75)

        st.caption(f"Kernel Size: {ksize}×{ksize} applied across all 4 spatial filters")
        c1, c2, c3, c4 = st.columns(4)
        for col, res_img, name in [
            (c1, f_box, "Averaging (Box)"),
            (c2, f_gauss, "Gaussian Blur"),
            (c3, f_median, "Median Filter"),
            (c4, f_bilateral, "Bilateral Filter")
        ]:
            with col:
                st.markdown(f'<div class="visualizer-card"><span class="card-tag tag-processed">{name}</span>', unsafe_allow_html=True)
                st.image(res_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 6: Image Inpainting (Restoration)
# ===========================================================================
def render_module_inpainting(image: np.ndarray):
    render_module_header(
        6,
        "Image Inpainting (Restoration)",
        "Restore corrupted, scratched, or watermarked image patches via boundary-driven partial differential equations: "
        "Alexandru Telea Fast Marching Method (`cv2.INPAINT_TELEA`) versus Navier-Stokes Fluid Dynamics (`cv2.INPAINT_NS`)."
    )

    h, w = image.shape[:2]
    c_mask_mode, c_rad = st.columns(2)
    with c_mask_mode:
        mask_source = st.selectbox(
            "Damage / Defect Mask Creation Mode:",
            [
                "Synthetic Scratches & Watermarks (Automated)",
                "Interactive Box Region Mask",
                "High-Brightness Threshold Mask",
                "Upload Custom Binary Mask"
            ]
        )
    with c_rad:
        inpaint_radius = st.slider("Inpainting Neighborhood Radius (px):", min_value=1, max_value=25, value=5)

    mask = np.zeros((h, w), dtype=np.uint8)
    damaged_image = image.copy()

    if mask_source == "Synthetic Scratches & Watermarks (Automated)":
        cv2.line(mask, (w // 8, h // 6), (3 * w // 4, 2 * h // 3), 255, 4)
        cv2.line(mask, (w // 4, 3 * h // 4), (5 * w // 6, h // 4), 255, 3)
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 10, 255, -1)
        cv2.putText(mask, "SCRATCH", (w // 3, h // 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
        damaged_image[mask > 0] = [255, 255, 255]

    elif mask_source == "Interactive Box Region Mask":
        bx = st.slider("Box Center X", 0, w, w // 2)
        by = st.slider("Box Center Y", 0, h, h // 2)
        bsize = st.slider("Box Dimension (px)", 10, min(h, w) // 2, 60)
        cv2.rectangle(mask, (max(0, bx - bsize // 2), max(0, by - bsize // 2)),
                      (min(w, bx + bsize // 2), min(h, by + bsize // 2)), 255, -1)
        damaged_image[mask > 0] = [255, 255, 255]

    elif mask_source == "High-Brightness Threshold Mask":
        gray = ensure_gray(image)
        thresh_val = st.slider("Detect Defective Pixels Above Cutoff:", 150, 255, 230)
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        damaged_image = image.copy()

    else:
        uploaded_mask = st.file_uploader("Upload Binary Mask Image (Damaged regions in white):", type=["png", "jpg", "jpeg"], key="inpaint_uploader")
        if uploaded_mask is not None:
            pil_mask = Image.open(uploaded_mask).convert('L').resize((w, h))
            mask = np.array(pil_mask, dtype=np.uint8)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            damaged_image[mask > 0] = [255, 255, 255]
        else:
            cv2.circle(mask, (w // 2, h // 2), 40, 255, -1)
            damaged_image[mask > 0] = [255, 255, 255]

    with st.spinner("Running Fast Marching (Telea) & Navier-Stokes restoration solvers..."):
        t0 = time.time()
        inpainted_telea = cv2.inpaint(damaged_image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        time_telea = (time.time() - t0) * 1000

        t1 = time.time()
        inpainted_ns = cv2.inpaint(damaged_image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
        time_ns = (time.time() - t1) * 1000

    c1, c2 = st.columns(2)
    with c1:
        overlay = damaged_image.copy()
        overlay[mask > 0] = [244, 63, 94]
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">CORRUPTED INPUT</span>', unsafe_allow_html=True)
        st.image(overlay, caption="Damaged Image with Target Mask (Red Overlay)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">BINARY INPAINT MASK</span>', unsafe_allow_html=True)
        st.image(mask, caption="Isolated Binary Restoration Mask", use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 🔬 Inpainted Restoration Output Comparison")
    col_tel, col_ns = st.columns(2)
    with col_tel:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">TELEA FAST MARCHING</span>', unsafe_allow_html=True)
        st.image(inpainted_telea, caption=f"cv2.INPAINT_TELEA ({time_telea:.1f}ms latency)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Processed Image (Telea)",
            data=to_pil_download_bytes(inpainted_telea),
            file_name="module6_telea_restored.png",
            mime="image/png"
        )
    with col_ns:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">NAVIER-STOKES FLUID DYNAMICS</span>', unsafe_allow_html=True)
        st.image(inpainted_ns, caption=f"cv2.INPAINT_NS ({time_ns:.1f}ms latency)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Processed Image (NS)",
            data=to_pil_download_bytes(inpainted_ns),
            file_name="module6_navier_stokes_restored.png",
            mime="image/png"
        )


# ===========================================================================
# MODULE 7: Lossless Compression Analysis
# ===========================================================================
def rle_encode(data: np.ndarray):
    """Encodes a flattened array using Run-Length Encoding (RLE)."""
    flat = data.flatten()
    n = len(flat)
    if n == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint32)
    change_idx = np.where(flat[:-1] != flat[1:])[0] + 1
    starts = np.concatenate(([0], change_idx))
    lengths = np.diff(np.concatenate((starts, [n])))
    values = flat[starts]
    return values, lengths


def rle_decode(values: np.ndarray, lengths: np.ndarray, shape: tuple) -> np.ndarray:
    """Decodes RLE runs back to the original image shape."""
    reconstructed = np.repeat(values, lengths)
    return reconstructed.reshape(shape)


def render_module_lossless_compression(image: np.ndarray):
    render_module_header(
        7,
        "Lossless Compression Analysis",
        "Evaluate storage footprint and information entropy using Run-Length Encoding (RLE), "
        "Lossless PNG (Deflate), and Lossless TIFF (LZW) containers with bit-exact reconstruction verification."
    )

    h, w = image.shape[:2]
    c = 1 if len(image.shape) == 2 else image.shape[2]
    raw_uncompressed_bytes = h * w * c
    raw_kb = raw_uncompressed_bytes / 1024.0

    gray = ensure_gray(image)

    png_compression_level = st.slider("PNG Deflate Compression Effort (0 = raw stream, 9 = maximum entropy)", 0, 9, 6)

    with st.spinner("Analyzing bitplanes and running lossless encoders..."):
        _, png_encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression_level])
        png_bytes = len(png_encoded.tobytes())
        png_kb = png_bytes / 1024.0

        pil_img = Image.fromarray(image)
        tiff_buf = io.BytesIO()
        pil_img.save(tiff_buf, format="TIFF", compression="tiff_lzw")
        tiff_bytes = len(tiff_buf.getvalue())
        tiff_kb = tiff_bytes / 1024.0

        rle_vals, rle_lens = rle_encode(gray)
        rle_encoded_bytes = len(rle_vals) + len(rle_lens) * 2
        rle_kb = rle_encoded_bytes / 1024.0

        decoded_rle = rle_decode(rle_vals, rle_lens, gray.shape)
        rle_max_diff = np.max(np.abs(gray.astype(int) - decoded_rle.astype(int)))
        is_rle_lossless = (rle_max_diff == 0)

    cr_png = raw_uncompressed_bytes / max(1, png_bytes)
    saved_png = (1.0 - png_bytes / raw_uncompressed_bytes) * 100.0

    cr_tiff = raw_uncompressed_bytes / max(1, tiff_bytes)
    saved_tiff = (1.0 - tiff_bytes / raw_uncompressed_bytes) * 100.0

    cr_rle = (h * w) / max(1, rle_encoded_bytes)
    saved_rle = (1.0 - rle_encoded_bytes / (h * w)) * 100.0

    st.markdown("#### 📈 Lossless Storage & Redundancy Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw Uncompressed Size", f"{raw_kb:.1f} KB", "Baseline 100%")
    with m2:
        st.metric("PNG (Deflate)", f"{png_kb:.1f} KB", f"{saved_png:.1f}% Saved ({cr_png:.2f}:1)")
    with m3:
        st.metric("TIFF (LZW)", f"{tiff_kb:.1f} KB", f"{saved_tiff:.1f}% Saved ({cr_tiff:.2f}:1)")
    with m4:
        st.metric("RLE (Grayscale Stream)", f"{rle_kb:.1f} KB", f"{saved_rle:.1f}% Saved ({cr_rle:.2f}:1)")

    if is_rle_lossless:
        st.success(f"✅ Lossless Integrity Verified: Bit-exact reproduction (Max pixel difference = {rle_max_diff}, MSE = 0.0).")
    else:
        st.error("❌ Lossless Integrity Check Failed.")

    configure_dark_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    labels = ['Raw Uncompressed', f'PNG (Level {png_compression_level})', 'TIFF (LZW)', 'RLE (Gray Runs)']
    sizes_kb = [raw_kb, png_kb, tiff_kb, rle_kb]
    colors = ['#64748B', '#38BDF8', '#10B981', '#F59E0B']

    bars = ax.barh(labels, sizes_kb, color=colors, height=0.5)
    for bar in bars:
        w_val = bar.get_width()
        ax.text(w_val + (max(sizes_kb) * 0.02), bar.get_y() + bar.get_height() / 2,
                f'{w_val:.1f} KB', va='center', fontweight='bold', color='#F8FAFC', fontsize=9)
    ax.set_xlabel("Payload Size (Kilobytes)", fontsize=10)
    ax.set_title("Lossless Compression Size Benchmark", fontsize=11, fontweight='600')
    ax.set_xlim([0, max(sizes_kb) * 1.2])
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    render_split_visualizer(image, decoded_rle, f"Reconstructed Lossless RLE Stream ({len(rle_vals)} runs)", "module7_rle_decoded.png")


# ===========================================================================
# MODULE 8: Morphological Operations
# ===========================================================================
def render_module_morphology(image: np.ndarray):
    render_module_header(
        8,
        "Morphological Operations",
        "Apply set-theoretic nonlinear mathematical morphology: Erosion, Dilation, Opening, Closing, "
        "and Morphological Gradient using customizable structuring elements."
    )

    c_op, c_shape, c_k, c_iter = st.columns(4)
    with c_op:
        operation = st.selectbox(
            "Morphological Operator:",
            ["Erosion (A ⊖ B)", "Dilation (A ⊕ B)", "Opening (A ∘ B)", "Closing (A • B)", "Morphological Gradient (Dilation - Erosion)"]
        )
    with c_shape:
        elem_shape_str = st.selectbox("Structuring Element Geometry:", ["RECTANGLE (MORPH_RECT)", "ELLIPSE (MORPH_ELLIPSE)", "CROSS (MORPH_CROSS)"])
    with c_k:
        ksize = st.slider("Structuring Element Kernel Size:", 1, 21, 5, step=2)
    with c_iter:
        iterations = st.slider("Successive Iterations:", 1, 10, 1)

    shape_map = {
        "RECTANGLE (MORPH_RECT)": cv2.MORPH_RECT,
        "ELLIPSE (MORPH_ELLIPSE)": cv2.MORPH_ELLIPSE,
        "CROSS (MORPH_CROSS)": cv2.MORPH_CROSS
    }
    kernel = cv2.getStructuringElement(shape_map[elem_shape_str], (ksize, ksize))

    with st.spinner("Executing morphological set operations..."):
        if "Erosion" in operation:
            result = cv2.erode(image, kernel, iterations=iterations)
            math_desc = r"A \ominus B = \{ z \mid (B)_z \subseteq A \}"
        elif "Dilation" in operation:
            result = cv2.dilate(image, kernel, iterations=iterations)
            math_desc = r"A \oplus B = \{ z \mid (\hat{B})_z \cap A \neq \emptyset \}"
        elif "Opening" in operation:
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            math_desc = r"A \circ B = (A \ominus B) \oplus B \quad \text{(Eliminates small foreground protrusions)}"
        elif "Closing" in operation:
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            math_desc = r"A \bullet B = (A \oplus B) \ominus B \quad \text{(Fills small holes and bridges gaps)}"
        else:
            result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
            math_desc = r"G(A) = (A \oplus B) - (A \ominus B) \quad \text{(Highlights structural boundaries)}"

    st.latex(math_desc)

    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">INPUT</span>', unsafe_allow_html=True)
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">MORPHOLOGICAL OUTPUT</span>', unsafe_allow_html=True)
        st.image(result, caption=f"Result: {operation.split('(')[0]}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Processed Image",
            data=to_pil_download_bytes(result),
            file_name="module8_morphology.png",
            mime="image/png"
        )
    with c3:
        st.markdown(f"**Structuring Element Matrix ({ksize}×{ksize}):**")
        configure_dark_matplotlib()
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(kernel, cmap='Blues', interpolation='nearest')
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, str(kernel[i, j]), ha='center', va='center',
                        color='white' if kernel[i, j] else '#64748B', fontsize=8, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ===========================================================================
# MODULE 9: Object Detection via Correlation
# ===========================================================================
def render_module_correlation(image: np.ndarray):
    render_module_header(
        9,
        "Object Detection via Correlation",
        "Localize target patterns across the scene using Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`). "
        "Crop a live region of interest or upload a template to isolate peak response coordinates."
    )

    h, w = image.shape[:2]
    c_source, c_method = st.columns(2)
    with c_source:
        template_source = st.radio("Template Selection Source:", ["Interactive Crop from Image", "Upload Template Image"], horizontal=True)
    with c_method:
        matching_method = st.selectbox(
            "Correlation Objective Metric:",
            ["cv2.TM_CCOEFF_NORMED (Normalized Cross-Correlation)", "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF_NORMED"]
        )

    method_dict = {
        "cv2.TM_CCOEFF_NORMED (Normalized Cross-Correlation)": cv2.TM_CCOEFF_NORMED,
        "cv2.TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "cv2.TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
    }
    match_flag = method_dict[matching_method]

    if template_source == "Interactive Crop from Image":
        col_crop1, col_crop2 = st.columns(2)
        with col_crop1:
            tx = st.slider("Template Anchor X", 0, max(1, w - 30), min(w // 4, max(1, w - 30)))
            ty = st.slider("Template Anchor Y", 0, max(1, h - 30), min(h // 4, max(1, h - 30)))
        with col_crop2:
            tw = st.slider("Template Width", 20, max(21, min(w - tx, 200)), min(65, max(21, min(w - tx, 200))))
            th = st.slider("Template Height", 20, max(21, min(h - ty, 200)), min(65, max(21, min(h - ty, 200))))

        template = image[ty:ty+th, tx:tx+tw].copy()
    else:
        uploaded_template = st.file_uploader("Upload Template Image (PNG/JPG):", type=["png", "jpg", "jpeg"], key="template_uploader")
        if uploaded_template is not None:
            pil_tpl = Image.open(uploaded_template).convert('RGB')
            template = np.array(pil_tpl)
            if template.shape[0] >= h or template.shape[1] >= w:
                st.warning("⚠️ Template larger than search frame; auto-scaling.")
                template = cv2.resize(template, (w // 3, h // 3))
        else:
            template = image[h//3:h//3+60, w//3:w//3+60].copy()

    img_gray = ensure_gray(image)
    tpl_gray = ensure_gray(template)
    th_h, th_w = tpl_gray.shape[:2]

    with st.spinner("Computing normalized cross-correlation surface..."):
        res_map = cv2.matchTemplate(img_gray, tpl_gray, match_flag)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_map)

    if match_flag == cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
        confidence = 1.0 - min_val
    else:
        top_left = max_loc
        confidence = max_val

    bottom_right = (top_left[0] + th_w, top_left[1] + th_h)

    detection_viz = image.copy()
    cv2.rectangle(detection_viz, top_left, bottom_right, (244, 63, 94), 3)
    cv2.circle(detection_viz, (top_left[0] + th_w // 2, top_left[1] + th_h // 2), 5, (16, 185, 129), -1)
    cv2.putText(
        detection_viz,
        f"Match: {confidence:.3f}",
        (top_left[0], max(16, top_left[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (244, 63, 94),
        2,
        cv2.LINE_AA
    )

    res_norm = cv2.normalize(res_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    res_heatmap = cv2.applyColorMap(res_norm, cv2.COLORMAP_JET)
    res_heatmap = cv2.cvtColor(res_heatmap, cv2.COLOR_BGR2RGB)

    st.latex(r"R(x,y) = \frac{\sum_{x',y'} (T'(x',y') \cdot I'(x+x', y+y'))}{\sqrt{\sum_{x',y'} T'^2(x',y') \cdot \sum_{x',y'} I'^2(x+x', y+y')}}")

    c1, c2, c3 = st.columns([1.8, 1, 1.8])
    with c1:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">LOCALIZED OBJECT</span>', unsafe_allow_html=True)
        st.image(detection_viz, caption=f"Detection Box (Confidence: {confidence:.3f})", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Processed Image",
            data=to_pil_download_bytes(detection_viz),
            file_name="module9_detection_result.png",
            mime="image/png"
        )
    with c2:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">QUERY TEMPLATE</span>', unsafe_allow_html=True)
        st.image(template, caption=f"Template ({th_w}×{th_h} px)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.metric("Peak Correlation", f"{confidence:.3f}")
        st.metric("Bounding Anchor", f"({top_left[0]}, {top_left[1]})")
    with c3:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">RESPONSE SURFACE</span>', unsafe_allow_html=True)
        st.image(res_heatmap, caption="Normalized Correlation Response (JET)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 10: Top-Hat Transformation
# ===========================================================================
def render_module_tophat(image: np.ndarray):
    render_module_header(
        10,
        "Top-Hat Transformation",
        "Extract subtle micro-structures and eliminate non-uniform background illumination gradients using "
        "**White Top-Hat** ($f - (f \\circ b)$) and **Black-Hat** ($(f \\bullet b) - f$) filters."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ksize = st.slider("Structuring Element Kernel Size (Odd)", 3, 51, 15, step=2)
    with c2:
        kernel_shape_name = st.selectbox("Kernel Geometry:", ["MORPH_RECT", "MORPH_ELLIPSE", "MORPH_CROSS"])
    with c3:
        contrast_boost = st.slider("Illumination Contrast Multiplier (γ)", 1.0, 3.0, 1.5, step=0.1)

    shape_map = {
        "MORPH_RECT": cv2.MORPH_RECT,
        "MORPH_ELLIPSE": cv2.MORPH_ELLIPSE,
        "MORPH_CROSS": cv2.MORPH_CROSS
    }
    kernel = cv2.getStructuringElement(shape_map[kernel_shape_name], (ksize, ksize))

    gray = ensure_gray(image)

    with st.spinner("Extracting Top-Hat and Black-Hat structural residual bitplanes..."):
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        enhanced = np.clip(gray.astype(np.float32) + contrast_boost * tophat.astype(np.float32) - contrast_boost * blackhat.astype(np.float32), 0, 255).astype(np.uint8)

    st.latex(r"T_{\text{hat}}(f) = f - (f \circ b) \quad\text{and}\quad B_{\text{hat}}(f) = (f \bullet b) - f")

    render_split_visualizer(gray, enhanced, f"Illumination & Contrast Corrected (γ = {contrast_boost})", "module10_enhanced.png", "Input Grayscale (Uneven Illumination)")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">WHITE TOP-HAT</span>', unsafe_allow_html=True)
        st.image(tophat, caption=f"Top-Hat (Bright Micro-Details, k={ksize})", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">BLACK-HAT</span>', unsafe_allow_html=True)
        st.image(blackhat, caption=f"Black-Hat (Dark Micro-Details, k={ksize})", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 11: Color Space Transformations
# ===========================================================================
def render_module_color_spaces(image: np.ndarray):
    render_module_header(
        11,
        "Color Space Transformations",
        "Convert RGB data into perceptual and industrial representations: **HSV** (Hue-Saturation-Value), "
        "**YCrCb** (Luma & Chroma), and **CIE LAB** (Perceptually Uniform Lightness & Color-Opponent axes)."
    )

    rgb = ensure_rgb(image)

    c_space, c_viz = st.columns(2)
    with c_space:
        color_space = st.selectbox("Target Color Space Representation:", ["HSV (Hue, Saturation, Value)", "YCrCb (Luminance, Chrominance)", "CIE LAB (Lightness, A*, B*)"])
    with c_viz:
        viz_style = st.radio("Channel Display Palette:", ["Perceptual Heatmap (Turbo/Viridis)", "Grayscale Intensity"], horizontal=True)

    with st.spinner("Splitting color spaces into individual orthogonal manifolds..."):
        if "HSV" in color_space:
            converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            ch1, ch2, ch3 = cv2.split(converted)
            names = ["Channel 0: Hue (0-179°)", "Channel 1: Saturation (0-255)", "Channel 2: Value (0-255)"]
            descriptions = [
                "Dominant spectral wavelength angle.",
                "Color saturation / chromatic purity.",
                "Radiant brightness / luminous intensity."
            ]
        elif "YCrCb" in color_space:
            converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            ch1, ch2, ch3 = cv2.split(converted)
            names = ["Channel 0: Y (Luminance)", "Channel 1: Cr (Red-Chroma)", "Channel 2: Cb (Blue-Chroma)"]
            descriptions = [
                "Achromatic luminance (perceptual luma).",
                "Chrominance red difference (R - Y offset).",
                "Chrominance blue difference (B - Y offset)."
            ]
        else:
            converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            ch1, ch2, ch3 = cv2.split(converted)
            names = ["Channel 0: L* (Lightness)", "Channel 1: a* (Green-Magenta)", "Channel 2: b* (Blue-Yellow)"]
            descriptions = [
                "Perceptually uniform human lightness.",
                "Color-opponent green to magenta axis.",
                "Color-opponent blue to yellow axis."
            ]

    st.markdown("#### 🌈 Split Channel Manifolds")
    col1, col2, col3 = st.columns(3)
    channels = [ch1, ch2, ch3]

    for idx, col in enumerate([col1, col2, col3]):
        with col:
            st.markdown(f'<div class="visualizer-card"><span class="card-tag tag-processed">{names[idx]}</span>', unsafe_allow_html=True)
            ch_data = channels[idx]
            if "Heatmap" in viz_style:
                colormap = cv2.COLORMAP_TWILIGHT if (idx == 0 and "HSV" in color_space) else cv2.COLORMAP_TURBO
                heatmap = cv2.applyColorMap(ch_data, colormap)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap, use_container_width=True)
            else:
                st.image(ch_data, use_container_width=True, clamp=True)
            st.caption(descriptions[idx])
            st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODULE 12: Edge Detection Comparison
# ===========================================================================
def render_module_edge_detection(image: np.ndarray):
    render_module_header(
        12,
        "Edge Detection Comparison",
        "Directly evaluate spatial gradient operators: **Canny Multi-Stage Detector** (optimal SNR with hysteresis), "
        "**Sobel Operator** (orthogonal Gaussian-smoothed derivatives), and **Prewitt Operator** (discrete difference kernels)."
    )

    gray = ensure_gray(image)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### ⚙️ Canny Parameters")
        canny_t1 = st.slider("Lower Threshold (T1)", 0, 255, 50)
        canny_t2 = st.slider("Upper Threshold (T2)", 0, 255, 150)
    with c2:
        st.markdown("##### ⚙️ Sobel Parameters")
        sobel_ksize = st.selectbox("Sobel Kernel Dimension:", [1, 3, 5, 7], index=1)
    with c3:
        st.markdown("##### ⚙️ Prewitt Parameters")
        prewitt_thresh = st.slider("Prewitt Edge Threshold (0 = Raw)", 0, 255, 60)

    with st.spinner("Computing Canny, Sobel, and Prewitt gradient maps..."):
        canny_edges = cv2.Canny(gray, canny_t1, canny_t2, apertureSize=3, L2gradient=True)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.clip(sobel_mag / np.max(sobel_mag + 1e-6) * 255, 0, 255).astype(np.uint8)

        prewitt_kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        prewitt_x = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, prewitt_kx)
        prewitt_y = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, prewitt_ky)
        prewitt_mag = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_mag = np.clip(prewitt_mag / np.max(prewitt_mag + 1e-6) * 255, 0, 255).astype(np.uint8)
        if prewitt_thresh > 0:
            _, prewitt_mag = cv2.threshold(prewitt_mag, prewitt_thresh, 255, cv2.THRESH_BINARY)

        canny_density = (np.count_nonzero(canny_edges) / canny_edges.size) * 100.0
        sobel_density = (np.count_nonzero(sobel_mag > 50) / sobel_mag.size) * 100.0
        prewitt_density = (np.count_nonzero(prewitt_mag > 50) / prewitt_mag.size) * 100.0

    st.markdown("#### 🔍 Multi-Algorithm Comparative Edge Mosaic")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-source">GRAYSCALE INPUT</span>', unsafe_allow_html=True)
        st.image(gray, caption="Baseline Grayscale", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">CANNY</span>', unsafe_allow_html=True)
        st.image(canny_edges, caption=f"Canny ({canny_density:.1f}% density)", use_container_width=True)
        st.download_button(
            label="⬇️ Download Processed Image",
            data=to_pil_download_bytes(canny_edges),
            file_name="module12_canny.png",
            mime="image/png"
        )
    with col3:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">SOBEL</span>', unsafe_allow_html=True)
        st.image(sobel_mag, caption=f"Sobel ({sobel_density:.1f}% density)", use_container_width=True)
        st.download_button(
            label="⬇️ Download Processed Image",
            data=to_pil_download_bytes(sobel_mag),
            file_name="module12_sobel.png",
            mime="image/png"
        )
    with col4:
        st.markdown('<div class="visualizer-card"><span class="card-tag tag-processed">PREWITT</span>', unsafe_allow_html=True)
        st.image(prewitt_mag, caption=f"Prewitt ({prewitt_density:.1f}% density)", use_container_width=True)
        st.download_button(
            label="⬇️ Download Processed Image",
            data=to_pil_download_bytes(prewitt_mag),
            file_name="module12_prewitt.png",
            mime="image/png"
        )


# ===========================================================================
# Application Entry Point & Sidebar Layout
# ===========================================================================
def main():
    # Hero SaaS Header Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badges-row">
            <div class="badge-pill-online"><span class="dot"></span> System Online</div>
            <div class="badge-pill-version">v2.0 Active</div>
            <div class="badge-pill-version">Accelerated CV2 Backend</div>
        </div>
        <h1 class="hero-title">VISION LAB // STUDIO</h1>
        <p class="hero-subtitle">Interactive Digital Image Processing & Computer Vision Engineering Workbench</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & Workspace Controls
    st.sidebar.markdown('<div class="sidebar-section-header">WORKBENCH NAVIGATION</div>', unsafe_allow_html=True)
    modules = [
        "1. Environment Setup & Library Overview",
        "2. Image Format Conversion & Bitwise Arithmetic",
        "3. 2D Geometric Transformations",
        "4. Spatial Enhancement & Segmentation",
        "5. Spatial Domain Filtering",
        "6. Image Inpainting (Restoration)",
        "7. Lossless Compression Analysis",
        "8. Morphological Operations",
        "9. Object Detection via Correlation",
        "10. Top-Hat Transformation",
        "11. Color Space Transformations",
        "12. Edge Detection Comparison"
    ]
    selected_module = st.sidebar.selectbox("Active Pipeline Module:", modules, index=0)

    st.sidebar.markdown('<div class="sidebar-section-header">IMAGE DATA BUFFER</div>', unsafe_allow_html=True)
    image_source = st.sidebar.radio(
        "Source Stream:",
        ["Built-in Sample Datasets", "Upload Custom Image (JPG/PNG)"]
    )

    if image_source == "Built-in Sample Datasets":
        sample_choice = st.sidebar.selectbox(
            "Sample Dataset Pattern:",
            [
                "Standard CV Test Pattern",
                "Damaged Photo (For Inpainting)",
                "Document Scan (Uneven Illumination)",
                "Geometric Coins & Particles"
            ],
            index=1 if "Inpainting" in selected_module else (2 if "Top-Hat" in selected_module else 0)
        )
        active_image = generate_sample_image(sample_choice)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image file:",
            type=["jpg", "jpeg", "png"],
            help="Supports JPG, JPEG, and PNG formats"
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            active_image = np.array(pil_image)
            st.sidebar.success("✨ Image loaded into memory buffer")
        else:
            st.sidebar.info("💡 Showing default test pattern until upload.")
            active_image = generate_sample_image("Standard CV Test Pattern")

    # Image Properties Expander
    with st.expander("📊 Active Image Properties & Sensor Metadata", expanded=False):
        h, w = active_image.shape[:2]
        c = 1 if len(active_image.shape) == 2 else active_image.shape[2]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Dimensions", f"{w} × {h} px")
        with c2:
            st.metric("Color Channels", f"{c} ({'Grayscale' if c == 1 else 'RGB'})")
        with c3:
            st.metric("Dynamic Range", f"[{int(np.min(active_image))}, {int(np.max(active_image))}]")
        with c4:
            st.metric("Intensity Mean ± Std", f"{float(np.mean(active_image)):.1f} ± {float(np.std(active_image)):.1f}")

    # Module Dispatcher
    if selected_module.startswith("1."):
        render_module_env_setup(active_image)
    elif selected_module.startswith("2."):
        render_module_format_bitwise(active_image)
    elif selected_module.startswith("3."):
        render_module_geometric(active_image)
    elif selected_module.startswith("4."):
        render_module_spatial_enhancement(active_image)
    elif selected_module.startswith("5."):
        render_module_spatial_filtering(active_image)
    elif selected_module.startswith("6."):
        render_module_inpainting(active_image)
    elif selected_module.startswith("7."):
        render_module_lossless_compression(active_image)
    elif selected_module.startswith("8."):
        render_module_morphology(active_image)
    elif selected_module.startswith("9."):
        render_module_correlation(active_image)
    elif selected_module.startswith("10."):
        render_module_tophat(active_image)
    elif selected_module.startswith("11."):
        render_module_color_spaces(active_image)
    elif selected_module.startswith("12."):
        render_module_edge_detection(active_image)


if __name__ == "__main__":
    main()
