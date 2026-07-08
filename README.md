# 🩸 Anemia FusionNet: Multimodal Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

## 📌 Project Overview
**Anemia FusionNet** is an advanced multimodal Artificial Intelligence framework designed to accurately predict anemia risk. Traditional diagnostic models often rely on a single source of data (e.g., only analyzing an eye image or only looking at a blood test). This project mimics a comprehensive clinical diagnosis by fusing three distinct data modalities using a **Transformer-based Multi-Head Attention** architecture.

**Author:** Vatsal Pujara  

---

## 🧠 System Architecture

The framework extracts and integrates features from three independent streams:

1. **Visual Modality (Conjunctiva Eye Images):** A custom 3-layer Convolutional Neural Network (CNN) extracts 64-dimensional visual feature vectors from palpebral conjunctiva images.
2. **Clinical Modality (CBC Blood Data):** A dense neural network processes 8 critical Complete Blood Count parameters (WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT).
3. **Geographical Modality (Regional Risk):** A geographic risk embedding module assigns baseline prevalence weights using state-wise statistics derived from the National Family Health Survey (NFHS-5).

### 🧬 Transformer Fusion Layer
Instead of standard vector concatenation, this network aligns the three modalities into a uniform sequence space. A **Multi-Head Attention** block dynamically weights the relationships between visual cues, clinical metrics, and regional risk, resulting in a highly robust, region-aware binary classification (Anemic vs. Non-Anemic).

---

## 📂 Repository Structure

```text
AnemiaFusionNet/
│
├── diagnosed_cbc_data_v4.csv    # Tabular clinical CBC records
├── indiadata.csv                # NFHS-5 State-wise anemia statistics
├── anemia.ipynb                 # Core notebook: Data synthesis, model building, and training
├── appAnemia.py                 # Streamlit web application interface
├── image_extractor.h5           # Saved weights for the custom CNN feature extractor
├── anemia_fusionnet.h5          # Saved weights for the final Multimodal Transformer
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
