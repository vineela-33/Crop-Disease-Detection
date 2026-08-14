import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/crop_disease_model.h5")
    with open("model/class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

st.markdown("""
<style>
/* Hide ALL streamlit default elements that look ugly */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stDeprecationWarning"] { display: none !important; }
.stAlert { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes floatDrop {
    0% { transform: translateY(-100px) rotate(0deg); opacity: 0; }
    50% { opacity: 0.35; }
    100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
}
@keyframes slideIn {
    0% { transform: translateY(20px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}
@keyframes pulseBorder {
    0%, 100% { border-color: #1F4D40; box-shadow: 0 0 8px rgba(16,185,129,0.2); }
    50% { border-color: #10B981; box-shadow: 0 0 20px rgba(16,185,129,0.5); }
}
@keyframes glowTitle {
    0%, 100% { filter: drop-shadow(0 0 15px #10B981); }
    50% { filter: drop-shadow(0 0 30px #A3E635) drop-shadow(0 0 60px #10B981); }
}

/* Main background */
.stApp {
    background: linear-gradient(160deg, #071A17 0%, #0D2923 50%, #071A17 100%) !important;
    color: #F0FDF4;
}

/* Remove default streamlit padding */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
}

/* Floating background */
.floating-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.emoji-item {
    position: absolute;
    animation: floatDrop linear infinite;
    opacity: 0.25;
}
.emoji-item:nth-child(1) { left: 5%;  font-size: 28px; animation-duration: 9s;  animation-delay: 0s; }
.emoji-item:nth-child(2) { left: 15%; font-size: 20px; animation-duration: 13s; animation-delay: 2s; }
.emoji-item:nth-child(3) { left: 28%; font-size: 33px; animation-duration: 10s; animation-delay: 4s; }
.emoji-item:nth-child(4) { left: 42%; font-size: 24px; animation-duration: 15s; animation-delay: 1s; }
.emoji-item:nth-child(5) { left: 58%; font-size: 30px; animation-duration: 11s; animation-delay: 3s; }
.emoji-item:nth-child(6) { left: 72%; font-size: 25px; animation-duration: 12s; animation-delay: 5s; }
.emoji-item:nth-child(7) { left: 85%; font-size: 35px; animation-duration: 8s;  animation-delay: 2s; }
.emoji-item:nth-child(8) { left: 93%; font-size: 27px; animation-duration: 14s; animation-delay: 6s; }

/* Title */
.title {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(90deg, #10B981, #A3E635, #34D399, #A3E635, #10B981);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite, glowTitle 3s ease-in-out infinite;
    font-family: Georgia, serif;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 5px;
    transition: transform 0.3s ease;
    position: relative;
    z-index: 10;
}
.title:hover { transform: scale(1.04); }

.subtitle {
    color: #A7C4B5;
    text-align: center;
    font-size: 17px;
    letter-spacing: 2px;
    position: relative;
    z-index: 10;
    margin-top: 5px;
}
.divider {
    width: 55%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #10B981, #A3E635, #10B981, transparent);
    margin: 12px auto 25px auto;
}

/* Section headers */
.section-header {
    color: #10B981 !important;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
    border-bottom: 1px solid #1F4D40;
    padding-bottom: 8px;
    margin-bottom: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071A17, #0D2923) !important;
    border-right: 1px solid #1F4D40 !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 20px !important;
}

/* Hide default file uploader — show only our custom one */
[data-testid="stFileUploader"] {
    background: #102E27 !important;
    border: 1.5px solid #10B981 !important;
    border-radius: 14px !important;
    animation: pulseBorder 3s infinite;
    transition: all 0.3s ease;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #A3E635 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16,185,129,0.3) !important;
}
[data-testid="stFileUploader"] label {
    color: #10B981 !important;
    font-weight: bold !important;
}
[data-testid="stFileDropzone"] {
    background: #0D2923 !important;
    border: 1.5px dashed #10B981 !important;
    border-radius: 10px !important;
    color: #10B981 !important;
}
/* Browse files button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #10B981, #34D399) !important;
    color: #071A17 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #A3E635, #10B981) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 0 15px rgba(163,230,53,0.4) !important;
}
/* Upload text color */
[data-testid="stFileDropzone"] p {
    color: #10B981 !important;
}
[data-testid="stFileDropzone"] small {
    color: #A7C4B5 !important;
}

/* Progress bar — FIXED */
[data-testid="stProgressBar"] > div {
    background-color: #1F4D40 !important;
    border-radius: 10px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #10B981, #A3E635) !important;
    border-radius: 10px !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #10B981, #A3E635) !important;
}

/* Result cards */
.result-box {
    border-radius: 18px;
    padding: 28px 20px;
    margin: 10px 0;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: slideIn 0.5s ease-out;
}
.result-box:hover { transform: translateY(-4px); }
.healthy {
    background: linear-gradient(135deg, #0a2a1a, #0D3D2B);
    border: 1.5px solid #10B981;
    box-shadow: 0 0 30px rgba(16,185,129,0.25);
}
.diseased {
    background: linear-gradient(135deg, #2D1515, #3D1A1A);
    border: 1.5px solid #F87171;
    box-shadow: 0 0 30px rgba(248,113,113,0.25);
}

/* Prediction cards */
.pred-card {
    background: #102E27;
    border: 1px solid #1F4D40;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    transition: all 0.2s ease;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pred-card:hover {
    border-color: #10B981;
    transform: translateX(4px);
    box-shadow: 0 4px 15px rgba(16,185,129,0.15);
}

/* Text */
p, li, label { color: #A7C4B5 !important; }
h1, h2, h3 { color: #10B981 !important; }
span { color: #A7C4B5; }

/* Image border */
img {
    border-radius: 12px !important;
    border: 1px solid #1F4D40 !important;
}
</style>

<div class="floating-bg">
    <div class="emoji-item">🌿</div>
    <div class="emoji-item">🍃</div>
    <div class="emoji-item">🌱</div>
    <div class="emoji-item">🌾</div>
    <div class="emoji-item">🌿</div>
    <div class="emoji-item">🍃</div>
    <div class="emoji-item">🌱</div>
    <div class="emoji-item">🌾</div>
</div>

<div style='padding: 35px 20px 5px 20px; position: relative; z-index: 10;'>
    <div class='title'>🌿 Crop Disease Detection</div>
    <p class='subtitle'>✨ AI-Powered Plant Disease Detection using Deep Learning ✨</p>
    <div class='divider'></div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='section-header'>🌱 About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#A7C4B5; font-size:14px; line-height:1.7;'>
    Uses <span style='color:#10B981; font-weight:bold;'>Transfer Learning</span>
    with <span style='color:#A3E635; font-weight:bold;'>MobileNetV2</span> to detect
    38 crop diseases with <span style='color:#FBBF24; font-weight:bold;'>93%+ accuracy!</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🌾 Supported Crops</div>", unsafe_allow_html=True)
    crops = ["🍎 Apple", "🍇 Grape", "🍅 Tomato", "🌽 Corn", "🥔 Potato",
         "🍑 Peach", "🌶️ Bell Pepper", "🍓 Strawberry", "🍒 Blueberry", "🍊 Orange", "And more!"]
    for crop in crops:
        st.markdown(f"<div style='color:#A7C4B5; padding:3px 0; font-size:13px;'>• {crop}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>⚙️ Model Info</div>", unsafe_allow_html=True)
    info_items = [
        ("🧠", "Model", "MobileNetV2"),
        ("📊", "Dataset", "87,000 images"),
        ("🎯", "Classes", "38 diseases"),
        ("✅", "Accuracy", "93%+"),
        ("⚡", "Method", "Transfer Learning"),
    ]
    for icon, label, value in info_items:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #1F4D40;'>
            <span style='color:#A7C4B5; font-size:13px;'>{icon} {label}</span>
            <span style='color:#10B981; font-size:13px; font-weight:bold;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:12px; background:#102E27; border:1px solid #1F4D40; border-radius:10px;'>
        <div style='color:#A7C4B5; font-size:11px;'>Built  by</div>
        <div style='color:#10B981; font-size:15px; font-weight:bold; margin-top:4px;'>Vineela Nagalla 🌿</div>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='section-header'>📸 Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload crop leaf",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Leaf Image", use_container_width=True)
        st.markdown(f"""
        <div style='background:#102E27; border:1px solid #1F4D40; border-radius:8px;
                    padding:10px 14px; margin-top:10px; display:flex; align-items:center; gap:8px;'>
            <span style='font-size:18px;'>📁</span>
            <span style='color:#A7C4B5; font-size:13px;'>File: </span>
            <span style='color:#10B981; font-size:13px; font-weight:bold;'>{uploaded_file.name}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#102E27; border:2px dashed #1F4D40; border-radius:16px;
                    padding:70px 20px; text-align:center; margin-top:10px;
                    transition: all 0.3s ease;'>
            <div style='font-size:55px; margin-bottom:15px;'>🌿</div>
            <div style='color:#A7C4B5; font-size:16px; font-weight:500;'>
                Drag & drop or browse a leaf image
            </div>
            <div style='color:#1F4D40; font-size:13px; margin-top:8px;'>
                Supports JPG, JPEG, PNG • Max 200MB
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-header'>🔍 Detection Result</div>", unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🤔 Analyzing leaf image..."):
            model, class_names = load_model()
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100
            is_healthy = "healthy" in predicted_class.lower()

        if is_healthy:
            st.markdown(f"""
            <div class='result-box healthy'>
                <div style='font-size:55px; margin-bottom:10px;'>✅</div>
                <div style='color:#10B981; font-size:28px; font-weight:900;
                            letter-spacing:2px; margin-bottom:8px;'>HEALTHY!</div>
                <div style='color:#34D399; font-size:16px; margin-bottom:12px;'>
                    {predicted_class.replace('_', ' ')}
                </div>
                <div style='background:rgba(16,185,129,0.1); border-radius:8px; padding:8px;'>
                    <span style='color:#A3E635; font-size:24px; font-weight:bold;'>
                        {confidence:.1f}% Confident
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class='result-box diseased'>
                <div style='font-size:55px; margin-bottom:10px;'>⚠️</div>
                <div style='color:#F87171; font-size:28px; font-weight:900;
                            letter-spacing:2px; margin-bottom:8px;'>DISEASE DETECTED!</div>
                <div style='color:#FCA5A5; font-size:16px; margin-bottom:12px;'>
                    {predicted_class.replace('_', ' ')}
                </div>
                <div style='background:rgba(248,113,113,0.1); border-radius:8px; padding:8px;'>
                    <span style='color:#FBBF24; font-size:24px; font-weight:bold;'>
                        {confidence:.1f}% Confident
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#A7C4B5; font-size:14px; font-weight:600; margin-bottom:6px;'>
            📊 Confidence Score
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence/100)

        # Top 3 predictions
        st.markdown("""
        <div style='color:#10B981; font-size:15px; font-weight:700;
                    margin-top:18px; margin-bottom:10px; letter-spacing:1px;'>
            🎯 Top 3 Predictions
        </div>
        """, unsafe_allow_html=True)

        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        medals = ["🥇", "🥈", "🥉"]
        colors = ["#A3E635", "#34D399", "#10B981"]

        for i, idx in enumerate(top3_idx):
            pct = predictions[0][idx]*100
            st.markdown(f"""
            <div class='pred-card'>
                <span style='color:#A7C4B5; font-size:14px;'>
                    {medals[i]} {class_names[idx].replace('_', ' ')}
                </span>
                <span style='color:{colors[i]}; font-size:14px; font-weight:bold;'>
                    {pct:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#102E27; border:1px solid #1F4D40; border-radius:16px;
                    padding:85px 20px; text-align:center;'>
            <div style='font-size:55px; margin-bottom:15px;'>🔍</div>
            <div style='color:#A7C4B5; font-size:16px; font-weight:500;'>
                Upload a leaf image to detect disease
            </div>
            <div style='color:#1F4D40; font-size:13px; margin-top:8px;'>
                Results will appear here instantly
            </div>
        </div>
        """, unsafe_allow_html=True)
