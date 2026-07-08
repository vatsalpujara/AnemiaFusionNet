import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anemia FusionNet", page_icon="🩸", layout="wide")
st.title("🩸 Anemia FusionNet: Multimodal Detection System")
st.markdown("Integrates Conjunctiva Images, Clinical CBC Data, and Geo-Risk factors for region-aware AI diagnosis.")

# --- LOAD MODELS (CACHED FOR SPEED) ---
@st.cache_resource
def load_ai_models():
    # Load your custom models saved from the Jupyter Notebook
    extractor = tf.keras.models.load_model('image_extractor.h5', compile=False)
    fusion = tf.keras.models.load_model('anemia_fusionnet.h5', compile=False)
    return extractor, fusion

try:
    image_extractor, fusion_model = load_ai_models()
    models_loaded = True
except Exception as e:
    st.error(f"Please ensure 'image_extractor.h5' and 'anemia_fusionnet.h5' are in the same folder. Error: {e}")
    models_loaded = False

# --- UI LAYOUT: THREE COLUMNS FOR THREE MODALITIES ---
col1, col2, col3 = st.columns(3)

# 1. IMAGE MODALITY
with col1:
    st.header("👁️ 1. Conjunctiva Image")
    uploaded_file = st.file_uploader("Upload Eye Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Patient Conjunctiva", use_column_width=True)

# 2. CLINICAL MODALITY
with col2:
    st.header("🩸 2. Clinical Data (CBC)")
    wbc = st.number_input("WBC (White Blood Cells)", value=6.5)
    rbc = st.number_input("RBC (Red Blood Cells)", value=4.5)
    hgb = st.number_input("HGB (Hemoglobin)", value=12.0)
    hct = st.number_input("HCT (Hematocrit)", value=36.0)
    mcv = st.number_input("MCV", value=80.0)
    mch = st.number_input("MCH", value=27.0)
    mchc = st.number_input("MCHC", value=33.0)
    plt = st.number_input("PLT (Platelets)", value=250.0)

# 3. GEO-RISK MODALITY
with col3:
    st.header("🗺️ 3. Geographic Risk")
    # Dictionary containing the NFHS-5 risk scores we generated in the notebook
    geo_dict = {
        'gujarat': 0.797, 'kerala': 0.255, 'assam': 0.650, 'maharashtra': 0.63, 
        'bihar': 0.68, 'karnataka': 0.55, 'delhi': 0.58, 'rajasthan': 0.61
    }
    selected_state = st.selectbox("Select Patient Location", [state.title() for state in geo_dict.keys()])
    geo_score = geo_dict[selected_state.lower()]
    st.info(f"Regional Risk Score for {selected_state}: **{geo_score:.3f}**")

# --- EXECUTE FUSION INFERENCE ---
st.markdown("---")
if st.button("🧬 Analyze Multimodal Data", use_container_width=True) and models_loaded:
    if uploaded_file is None:
        st.warning("Please upload an image first!")
    else:
        with st.spinner("Processing through Transformer FusionNet..."):
            # 1. Process Image
            img_resized = img.resize((224, 224))
            img_arr = img_to_array(img_resized) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            img_features = image_extractor.predict(img_arr, verbose=0)
            
            # 2. Process Clinical (Raw inputs - in production, apply standard scaling here)
            clin_features = np.array([[wbc, rbc, hgb, hct, mcv, mch, mchc, plt]], dtype='float32')
            
            # 3. Process Geo
            geo_features = np.array([[geo_score]], dtype='float32')
            
            # 4. Final Fusion Prediction
            prediction = fusion_model.predict([img_features, clin_features, geo_features], verbose=0)[0][0]
            
            # --- DISPLAY RESULTS ---
            st.markdown("### 📊 Diagnostic Result")
            if prediction > 0.5:
                st.error(f"**High Risk of Anemia Detected** (Confidence: {prediction*100:.1f}%)")
            else:
                st.success(f"**Low Risk of Anemia** (Confidence: {(1-prediction)*100:.1f}%)")