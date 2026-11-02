
# import streamlit as st
# import tensorflow as tf
# import numpy as np

# from pathlib import Path
# from PIL import Image, ImageOps


# # =====================================================
# # 1. CONFIGURATION
# # =====================================================

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = BASE_DIR / "models" / "waste_classifier_finetuned.keras"
# CLASSES_PATH = BASE_DIR / "models" / "classes.txt"

# DEFAULT_CLASSES = [
#     "cardboard",
#     "glass",
#     "metal",
#     "paper",
#     "plastic",
#     "trash"
# ]

# IMG_SIZE = (224, 224)
# CONFIDENCE_THRESHOLD = 0.70

# CATEGORY_ICONS = {
#     "cardboard": "📦",
#     "glass": "🍾",
#     "metal": "🥫",
#     "paper": "📄",
#     "plastic": "🧴",
#     "trash": "🗑️"
# }


# # =====================================================
# # 2. PAGE CONFIGURATION
# # =====================================================

# st.set_page_config(
#     page_title="EcoVision AI | Waste Classifier",
#     page_icon="♻️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # =====================================================
# # 3. CUSTOM CSS
# # =====================================================

# st.markdown("""
# <style>

# .stApp {
#     background: #08111f;
#     color: #f1f5f9;
# }

# header[data-testid="stHeader"] {
#     background: transparent;
# }

# .block-container {
#     max-width: 1250px;
#     padding-top: 2rem;
#     padding-bottom: 3rem;
# }

# section[data-testid="stSidebar"] {
#     background: #0d1929;
#     border-right: 1px solid #26384d;
# }

# .hero-title {
#     font-size: 36px;
#     font-weight: 850;
#     color: #f8fafc;
#     letter-spacing: -1px;
# }

# .gradient-text {
#     background: linear-gradient(90deg, #34d399, #06b6d4);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }

# .subtitle {
#     color: #94a3b8;
#     font-size: 15px;
#     margin-top: 8px;
#     margin-bottom: 28px;
# }

# .custom-card {
#     background: linear-gradient(145deg, #14283e, #0e1b2c);
#     border: 1px solid #294158;
#     border-radius: 16px;
#     padding: 22px;
#     margin-bottom: 16px;
# }

# .card-label {
#     color: #94a3b8;
#     font-size: 12px;
#     text-transform: uppercase;
#     letter-spacing: 1px;
# }

# .card-value {
#     font-size: 26px;
#     font-weight: 800;
#     color: #f8fafc;
#     margin-top: 8px;
# }

# .result-card {
#     background: linear-gradient(135deg, #103c32, #102c39);
#     border: 1px solid #218767;
#     border-radius: 18px;
#     padding: 26px;
#     margin: 14px 0;
# }

# .unknown-card {
#     background: linear-gradient(135deg, #422c19, #292033);
#     border-color: #b7791f;
# }

# .result-label {
#     color: #a7f3d0;
#     font-size: 12px;
#     text-transform: uppercase;
#     letter-spacing: 1.5px;
# }

# .result-class {
#     font-size: 32px;
#     font-weight: 850;
#     color: white;
#     margin: 10px 0;
# }

# .section-title {
#     font-size: 21px;
#     font-weight: 750;
#     color: #f1f5f9;
#     margin: 22px 0 14px 0;
# }

# div[data-testid="stFileUploader"] {
#     background: #102037;
#     border: 1px dashed #3b82a0;
#     border-radius: 14px;
#     padding: 12px;
# }

# div[data-testid="stProgressBar"] > div {
#     background: #20364c;
# }

# div[data-testid="stProgressBar"] > div > div {
#     background: linear-gradient(90deg, #10b981, #06b6d4);
# }

# .stButton button {
#     background: linear-gradient(90deg, #059669, #0d9488);
#     color: white;
#     border: none;
#     border-radius: 10px;
#     font-weight: 700;
# }

# .footer {
#     text-align: center;
#     color: #64748b;
#     font-size: 12px;
#     padding: 25px 0;
#     margin-top: 40px;
#     border-top: 1px solid #26384d;
# }

# @media (max-width: 768px) {
#     .hero-title {
#         font-size: 27px;
#     }

#     .result-class {
#         font-size: 25px;
#     }
# }

# </style>
# """, unsafe_allow_html=True)


# # =====================================================
# # 4. LOAD CLASS NAMES
# # =====================================================

# def load_class_names():

#     if CLASSES_PATH.exists():

#         with open(CLASSES_PATH, "r", encoding="utf-8") as file:
#             classes = [
#                 line.strip()
#                 for line in file
#                 if line.strip()
#             ]

#         return classes

#     return DEFAULT_CLASSES


# CLASS_NAMES = load_class_names()


# # =====================================================
# # 5. LOAD MODEL
# # =====================================================

# @st.cache_resource
# def load_model():

#     return tf.keras.models.load_model(
#         str(MODEL_PATH),
#         compile=False
#     )


# if not MODEL_PATH.is_file():

#     st.error(f"Model file not found: {MODEL_PATH}")
#     st.stop()


# if not CLASS_NAMES:

#     st.error("No class names found.")
#     st.stop()


# try:

#     with st.spinner("Loading AI classification model..."):
#         model = load_model()

# except Exception as error:

#     st.error(f"Model loading failed: {error}")
#     st.stop()


# # =====================================================
# # 6. MODEL VALIDATION
# # =====================================================

# if model.output_shape[-1] != len(CLASS_NAMES):

#     st.error(
#         f"Model has {model.output_shape[-1]} outputs, "
#         f"but {len(CLASS_NAMES)} class names were found."
#     )

#     st.stop()


# # =====================================================
# # 7. SIDEBAR
# # =====================================================

# with st.sidebar:

#     st.markdown("""
#     <div style="padding:10px 0 25px;">

#         <div style="font-size:36px;">♻️</div>

#         <div style="
#             color:#34d399;
#             font-size:24px;
#             font-weight:850;">
#             EcoVision AI
#         </div>

#         <div style="color:#94a3b8;font-size:12px;">
#             Smart Waste Intelligence
#         </div>

#     </div>
#     """, unsafe_allow_html=True)

#     st.divider()

#     st.markdown("### AI Model Information")

#     st.write("**Architecture:** MobileNetV2")
#     st.write("**Framework:** TensorFlow")
#     st.write("**Input:** 224 × 224")
#     st.write("**Categories:** 6")
#     st.write("**Validation Accuracy:** 86.14%")

#     st.divider()

#     st.markdown("### Supported Categories")

#     for category in CLASS_NAMES:

#         icon = CATEGORY_ICONS.get(category, "♻️")

#         st.write(f"{icon} {category.capitalize()}")

#     st.divider()

#     st.caption("AI-Powered Waste Classification System")


# # =====================================================
# # 8. HEADER
# # =====================================================

# st.markdown("""
# <div>

#     <div class="hero-title">
#         AI Waste <span class="gradient-text">
#         Classification System
#         </span>
#     </div>

#     <div class="subtitle">
#         Intelligent waste recognition powered by
#         MobileNetV2 Deep Learning.
#         Identify and classify waste materials instantly.
#     </div>

# </div>
# """, unsafe_allow_html=True)


# # =====================================================
# # 9. DASHBOARD METRICS
# # =====================================================

# col1, col2, col3 = st.columns(3)

# with col1:

#     st.markdown("""
#     <div class="custom-card">
#         <div class="card-label">AI Architecture</div>
#         <div class="card-value">MobileNetV2</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:

#     st.markdown("""
#     <div class="custom-card">
#         <div class="card-label">Waste Categories</div>
#         <div class="card-value">06 Classes</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:

#     st.markdown("""
#     <div class="custom-card">
#         <div class="card-label">Validation Accuracy</div>
#         <div class="card-value">86.14%</div>
#     </div>
#     """, unsafe_allow_html=True)


# # =====================================================
# # 10. IMAGE UPLOAD
# # =====================================================

# st.markdown(
#     '<div class="section-title">📸 Upload Waste Image</div>',
#     unsafe_allow_html=True
# )

# st.caption(
#     "Upload a clear photograph of cardboard, glass, metal, "
#     "paper, plastic or trash."
# )

# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png", "webp"]
# )


# # =====================================================
# # 11. IMAGE CLASSIFICATION
# # =====================================================

# if uploaded_file is not None:

#     try:

#         # Open uploaded image
#         input_image = Image.open(uploaded_file)

#         # Correct camera orientation
#         input_image = ImageOps.exif_transpose(input_image).convert("RGB")

#         # Display columns
#         image_col, result_col = st.columns(
#             [1, 1],
#             gap="large"
#         )

#         with image_col:

#             st.markdown(
#                 '<div class="section-title">Uploaded Image</div>',
#                 unsafe_allow_html=True
#             )

#             st.image(
#                 input_image,
#                 use_container_width=True
#             )

#             st.caption(
#                 f"Image size: {input_image.width} × {input_image.height}"
#             )


#         # ---------------------------------------------
#         # PREPROCESSING
#         # ---------------------------------------------

#         resized_image = input_image.resize(IMG_SIZE)

#         image_array = np.array(
#             resized_image,
#             dtype=np.float32
#         )

#         image_array = np.expand_dims(
#             image_array,
#             axis=0
#         )


#         # ---------------------------------------------
#         # PREDICTION
#         # ---------------------------------------------

#         with st.spinner("AI is analyzing the image..."):

#             predictions = model.predict(
#                 image_array,
#                 verbose=0
#             )[0]

#         if not np.all(np.isfinite(predictions)):

#             st.error("Invalid prediction values returned by the model.")
#             st.stop()

#         if len(predictions) != len(CLASS_NAMES):

#             st.error("Prediction output does not match class names.")
#             st.stop()

#         predicted_index = int(np.argmax(predictions))

#         raw_predicted_class = CLASS_NAMES[predicted_index]

#         confidence = float(predictions[predicted_index])

#         is_unknown = confidence < CONFIDENCE_THRESHOLD

#         final_class = (
#             "unknown"
#             if is_unknown
#             else raw_predicted_class
#         )


#         # ---------------------------------------------
#         # RESULT CARD
#         # ---------------------------------------------

#         with result_col:

#             st.markdown(
#                 '<div class="section-title">🔍 AI Prediction</div>',
#                 unsafe_allow_html=True
#             )

#             if is_unknown:

#                 st.markdown(f"""
#                 <div class="result-card unknown-card">

#                     <div class="result-label">
#                         ⚠️ Low Confidence
#                     </div>

#                     <div class="result-class">
#                         UNKNOWN
#                     </div>

#                     <div style="color:#fde68a;">
#                         Confidence: {confidence * 100:.2f}%
#                     </div>

#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.warning(
#                     "The prediction is below the confidence threshold."
#                 )

#             else:

#                 icon = CATEGORY_ICONS.get(
#                     final_class,
#                     "♻️"
#                 )

#                 st.markdown(f"""
#                 <div class="result-card">

#                     <div class="result-label">
#                         ✓ Classification Successful
#                     </div>

#                     <div class="result-class">
#                         {icon} {final_class.upper()}
#                     </div>

#                     <div style="color:#d1fae5;">
#                         Confidence: {confidence * 100:.2f}%
#                     </div>

#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.success(
#                     f"Detected: {final_class.upper()}"
#                 )

#             st.metric(
#                 "Prediction Confidence",
#                 f"{confidence * 100:.2f}%"
#             )

#             st.progress(confidence)

#             st.caption(
#                 f"Raw model prediction: "
#                 f"{raw_predicted_class.upper()}"
#             )


#         # =================================================
#         # 12. CLASS PROBABILITIES
#         # =================================================

#         st.divider()

#         st.markdown(
#             '<div class="section-title">'
#             '📊 Class Probability Analysis'
#             '</div>',
#             unsafe_allow_html=True
#         )

#         st.caption(
#             "Probability scores generated by the trained model."
#         )

#         probability_col, details_col = st.columns(
#             [1.3, 1],
#             gap="large"
#         )

#         sorted_indices = np.argsort(predictions)[::-1]

#         with probability_col:

#             for index in sorted_indices:

#                 class_name = CLASS_NAMES[index]

#                 probability = float(predictions[index])

#                 percentage = probability * 100

#                 icon = CATEGORY_ICONS.get(
#                     class_name,
#                     "♻️"
#                 )

#                 st.markdown(
#                     f"""
#                     <div style="
#                         display:flex;
#                         justify-content:space-between;
#                         margin-top:13px;
#                         margin-bottom:5px;
#                     ">

#                         <span style="color:#e2e8f0;font-weight:600;">
#                             {icon} {class_name.capitalize()}
#                         </span>

#                         <span style="color:#34d399;font-weight:700;">
#                             {percentage:.2f}%
#                         </span>

#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#                 st.progress(
#                     min(max(probability, 0.0), 1.0)
#                 )


#         # =================================================
#         # 13. PREDICTION DETAILS
#         # =================================================

#         with details_col:

#             st.markdown(
#                 '<div class="section-title">'
#                 'Prediction Details'
#                 '</div>',
#                 unsafe_allow_html=True
#             )

#             st.write(
#                 "**Predicted category:**",
#                 raw_predicted_class.capitalize()
#             )

#             st.write(
#                 "**Prediction status:**",
#                 "Unknown" if is_unknown else "Accepted"
#             )

#             st.write(
#                 "**Confidence threshold:**",
#                 f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
#             )

#             top_two = sorted_indices[:2]

#             st.write(
#                 "**Second prediction:**",
#                 CLASS_NAMES[top_two[1]].capitalize()
#             )

#             st.write("**Model:** MobileNetV2")


#         # =================================================
#         # 14. FINAL SUMMARY
#         # =================================================

#         st.divider()

#         if not is_unknown:

#             st.markdown(f"""
#             <div class="custom-card">

#                 <div class="card-label">
#                     Final Classification
#                 </div>

#                 <div class="card-value gradient-text">
#                     {final_class.upper()}
#                 </div>

#                 <p style="color:#94a3b8;">
#                     The AI model assigned this image
#                     to the {final_class} category.
#                 </p>

#             </div>
#             """, unsafe_allow_html=True)

#         else:

#             st.info(
#                 "Please upload another clear image for a more "
#                 "reliable classification."
#             )


#     except Exception as error:

#         st.error(
#             f"Image classification failed: {error}"
#         )


# else:

#     # =================================================
#     # 15. EMPTY STATE
#     # =================================================

#     st.markdown("""
#     <div class="custom-card"
#          style="text-align:center;padding:45px 20px;">

#         <div style="font-size:55px;">♻️</div>

#         <h2 style="color:#f1f5f9;">
#             Ready to Classify Waste?
#         </h2>

#         <p style="color:#94a3b8;">
#             Upload an image to start AI-powered
#             waste classification.
#         </p>

#         <p style="
#             color:#34d399;
#             font-weight:700;
#             font-size:14px;
#         ">
#             Cardboard · Glass · Metal · Paper · Plastic · Trash
#         </p>

#     </div>
#     """, unsafe_allow_html=True)


# # =====================================================
# # 16. FOOTER
# # =====================================================

# st.markdown("""
# <div class="footer">

#     <div style="
#         color:#34d399;
#         font-size:18px;
#         font-weight:800;
#         margin-bottom:8px;
#     ">
#         ♻️ EcoVision AI
#     </div>

#     AI-Powered Waste Classification System

#     <br><br>

#     Powered by TensorFlow · Keras · MobileNetV2

#     <br><br>

#     Intelligent Technology for a Sustainable Future

# </div>
# """, unsafe_allow_html=True)


import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\Jetson-Nano_AIhardware Project\models\waste_classifier_finetuned.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# CHECK MODEL FILE
# ==========================================

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()


# ==========================================
# CHECK MODEL OUTPUTS
# ==========================================

if model.output_shape[-1] != len(CLASS_NAMES):
    st.error(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
    )
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.title("♻️ AI Waste Classification System")

st.write(
    "Upload an image. The AI model will classify it into one of six "
    "waste categories. Low-confidence predictions are marked as unknown."
)

st.info(
    f"Unknown threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%"
)

st.divider()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    input_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        input_image,
        caption="Uploaded Image"
    )

    # Resize image
    resized_image = input_image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = int(np.argmax(predictions))
    raw_predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    # Apply threshold
    if confidence < CONFIDENCE_THRESHOLD:
        final_class = "unknown"
        is_unknown = True
    else:
        final_class = raw_predicted_class
        is_unknown = False

    # ======================================
    # RESULT
    # ======================================

    st.divider()
    st.subheader("🔍 Classification Result")

    if is_unknown:
        st.warning(
            f"Detected: UNKNOWN\n\n"
            f"The highest model confidence was "
            f"{confidence * 100:.2f}%, which is below the "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold."
        )
    else:
        st.success(
            f"Detected: {final_class.upper()}"
        )

    st.metric(
        "Final Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.caption(
        f"Raw model prediction: {raw_predicted_class.upper()}"
    )

    # ======================================
    # CLASS PROBABILITIES
    # ======================================

    st.subheader("Class Probabilities")

    for class_name, probability in zip(CLASS_NAMES, predictions):
        probability = float(probability)

        st.write(
            f"{class_name.capitalize()}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)