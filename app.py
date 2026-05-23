import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from model import load_trained_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ParasiteVisionAI",
    page_icon="🦠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ================= BACKGROUND ================= */

.stApp {
    background: linear-gradient(
        135deg,
        #02111d 0%,
        #06263a 45%,
        #0b4666 100%
    );
    color: white;
}

/* Hide Streamlit default menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ================= TITLE ================= */

.main-title {
    font-size: 72px;
    font-weight: 900;
    text-align: center;
    color: #b7ff4a;
    margin-top: -20px;

    text-shadow:
        0px 0px 15px rgba(183,255,74,0.5),
        0px 0px 35px rgba(183,255,74,0.2);
}

.subtitle {
    text-align: center;
    font-size: 30px;
    font-weight: 600;
    color: white;
    margin-bottom: 40px;
}

/* ================= GLASS PANELS ================= */

.glass {
    background: rgba(0,0,0,0.45);
    border-radius: 25px;
    padding: 28px;

    border: 1px solid rgba(255,255,255,0.12);

    backdrop-filter: blur(10px);

    box-shadow:
        0px 8px 30px rgba(0,0,0,0.45);
}

/* ================= TEXT ================= */

h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

p, span, label, div {
    color: white !important;
    font-size: 18px;
}

/* ================= PREDICTION ================= */

.prediction {
    color: #9eff57;
    font-size: 38px;
    font-weight: bold;
    margin-top: 10px;
}

.confidence {
    color: white;
    font-size: 24px;
    margin-bottom: 15px;
}

/* ================= BUTTONS ================= */

.stButton > button {

    width: 100%;

    background: linear-gradient(
        to right,
        #19c37d,
        #11998e
    );

    color: white !important;

    font-size: 22px;
    font-weight: bold;

    border-radius: 15px;

    border: none;

    padding: 14px;

    transition: 0.3s ease;

    box-shadow:
        0px 5px 18px rgba(25,195,125,0.3);
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* ================= UPLOADER ================= */

[data-testid="stFileUploader"] {

    background: rgba(0, 120, 255, 0.25);

    border: 3px solid #00ffcc;

    border-radius: 22px;

    padding: 20px;

    box-shadow:
        0px 0px 20px rgba(0,255,200,0.35);

    backdrop-filter: blur(10px);
}

/* GREEN UPLOAD BUTTON */

[data-testid="stFileUploader"] button {

    background: #00c853 !important;

    color: white !important;

    font-size: 18px !important;

    font-weight: bold !important;

    border-radius: 14px !important;

    border: none !important;

    padding: 12px 24px !important;

    box-shadow:
        0px 0px 15px rgba(0,255,100,0.45) !important;
}

/* Drag area */

[data-testid="stFileUploaderDropzone"] {

    background: rgba(0,0,0,0.25) !important;

    border: 2px dashed #00ffcc !important;

    border-radius: 18px !important;
}

/* ================= CAMERA ================= */

[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 10px;
}

/* ================= FEATURE CARDS ================= */

.feature-card {

    background: rgba(0,0,0,0.35);

    padding: 20px;

    border-radius: 18px;

    text-align: center;

    font-size: 24px;
    font-weight: bold;

    color: white;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0px 5px 15px rgba(0,0,0,0.35);
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    margin-top: 50px;
    color: white;
    font-size: 18px;
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_PATH = "parasite_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model, label_map = load_trained_model(
        MODEL_PATH,
        DEVICE
    )

    return model, label_map

model, LABEL_MAP = load_model()

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# PARASITE INFORMATION DATABASE
# =========================================================

parasite_info = {

    "Ascaris lumbricoides": {
        "description":
        "A large intestinal roundworm infection transmitted through contaminated food and water.",

        "diagnosis":
        "Diagnosed using stool microscopy to detect eggs.",

        "prevention":
        "Wash hands, clean vegetables, and avoid contaminated water."
    },

    "Capillaria philippinensis": {
        "description":
        "An intestinal parasite causing diarrhea and malnutrition.",

        "diagnosis":
        "Detected through stool examination.",

        "prevention":
        "Avoid raw freshwater fish."
    },

    "Enterobius vermicularis": {
        "description":
        "Common pinworm infection causing anal itching.",

        "diagnosis":
        "Tape test or stool examination.",

        "prevention":
        "Frequent hand washing and sanitation."
    },

    "Fasciolopsis buski": {
        "description":
        "Large intestinal fluke transmitted by aquatic plants.",

        "diagnosis":
        "Stool microscopy for egg detection.",

        "prevention":
        "Avoid eating raw aquatic plants."
    },

    "Hookworm egg": {
        "description":
        "Intestinal worm causing anemia and weakness.",

        "diagnosis":
        "Microscopic stool examination.",

        "prevention":
        "Wear shoes and improve sanitation."
    },

    "Hymenolepis diminuta": {
        "description":
        "Rodent-associated tapeworm infection.",

        "diagnosis":
        "Detected using stool analysis.",

        "prevention":
        "Control rodents and food contamination."
    },

    "Hymenolepis nana": {
        "description":
        "Dwarf tapeworm common in children.",

        "diagnosis":
        "Stool examination.",

        "prevention":
        "Maintain hygiene and sanitation."
    },

    "Opisthorchis viverrine": {
        "description":
        "Liver fluke associated with bile duct disease.",

        "diagnosis":
        "Detected by stool microscopy.",

        "prevention":
        "Avoid undercooked freshwater fish."
    },

    "Paragonimus spp": {
        "description":
        "Lung fluke affecting the respiratory system.",

        "diagnosis":
        "Sputum or stool examination.",

        "prevention":
        "Cook crabs and crayfish thoroughly."
    },

    "Taenia spp. egg": {
        "description":
        "Tapeworm infection from undercooked meat.",

        "diagnosis":
        "Stool examination for eggs.",

        "prevention":
        "Cook meat properly."
    },

    "Trichuris trichiura": {
        "description":
        "Whipworm infection affecting the large intestine.",

        "diagnosis":
        "Detected using stool microscopy.",

        "prevention":
        "Use clean water and maintain sanitation."
    }
}

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    label = LABEL_MAP[predicted.item()]

    conf = confidence.item() * 100

    return label, conf, probs[0]

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🦠 ParasiteVisionAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Parasite Detection</div>',
    unsafe_allow_html=True
)

# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True
    )

    st.header("📤 Upload Sample Image")

    uploaded_file = st.file_uploader(
        "Upload parasite microscopic image",
        type=["jpg", "jpeg", "png"]
    )

    camera_image = st.camera_input(
        "📷 Capture from Camera"
    )

    analyze = st.button("🔍 Analyze")

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True
    )

    st.header("🧠 Results")

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

    elif camera_image:
        image = Image.open(camera_image).convert("RGB")

    if image and analyze:

        st.image(image, use_container_width=True)

        with st.spinner(
            "Analyzing parasite image..."
        ):

            label, conf, probs = predict_image(image)

        st.markdown(
            f'<div class="prediction">{label}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">Confidence: {conf:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(min(int(conf), 100))

        if label in parasite_info:

            info = parasite_info[label]

            st.markdown("---")

            st.subheader("🦠 Parasite Information")

            st.markdown("### Definition")
            st.write(info["description"])

            st.markdown("### Diagnosis")
            st.write(info["diagnosis"])

            st.markdown("### Prevention")
            st.write(info["prevention"])

        st.subheader("📊 Other Possible Matches")

        probs = probs.cpu().numpy()

        top_indices = probs.argsort()[-5:][::-1]

        data = []

        for idx in top_indices:

            data.append({
                "Parasite": LABEL_MAP[idx],
                "Confidence":
                f"{probs[idx]*100:.2f}%"
            })

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader("📝 Feedback")

        feedback = st.text_area(
            "If prediction is incorrect, report it here:"
        )

        if st.button("Submit Feedback"):

            os.makedirs("feedback", exist_ok=True)

            with open(
                "feedback/feedback.txt",
                "a"
            ) as f:

                f.write(
                    f"\nPrediction: {label} ({conf:.2f}%)\n"
                )

                f.write(
                    f"Feedback: {feedback}\n"
                )

                f.write("="*50 + "\n")

            st.success(
                "Feedback saved successfully."
            )

    else:

        st.info(
            "Upload or capture an image and click Analyze"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FEATURE CARDS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)

with f1:

    st.markdown(
        '<div class="feature-card">⚡ Fast Analysis</div>',
        unsafe_allow_html=True
    )

with f2:

    st.markdown(
        '<div class="feature-card">🎯 High Accuracy</div>',
        unsafe_allow_html=True
    )

with f3:

    st.markdown(
        '<div class="feature-card">📄 Detailed Report</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">Developed by Bisrat Weldegiyorgis</div>',
    unsafe_allow_html=True
)
