# app.py — AI-Based Visa Recommendation System (Purity & Destiny Theme)
# Developed by Team (6 Members)
# Bilingual Version — English + Tamil

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load trained model & encoder
# --------------------------
model = joblib.load("best_visa_model.pkl")     # Replace with your model file
label_encoder = joblib.load("label_encoder.pkl")

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="VIRA • Visa Recommendation System",
    page_icon="🌐",
    layout="centered"
)

# --------------------------
# Custom Styling
# --------------------------
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
        }
        .title {
            font-size: 38px;
            font-weight: 700;
            color: #1e293b;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #475569;
            text-align: center;
            margin-bottom: 30px;
        }
        .visa {
            background: linear-gradient(135deg, #22d3ee, #3b82f6);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌐 VIRA — AI-Based Visa Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering Smart Visa Choices for Global Applicants | உலகளாவிய விண்ணப்பதாரர்களுக்கான புத்திசாலி விசா பரிந்துரைகள்</div>', unsafe_allow_html=True)

# --------------------------
# Input Section
# --------------------------
st.sidebar.header("🧾 Applicant Details / விண்ணப்பதாரர் விவரங்கள்")

age = st.sidebar.slider("Applicant Age / விண்ணப்பதாரரின் வயது", 18, 70, 25)
education = st.sidebar.selectbox("Education Level / கல்வி நிலை", ["High School", "Diploma", "Bachelor's", "Master's", "PhD"])
occupation = st.sidebar.selectbox("Occupation / தொழில்", ["Student", "Engineer", "Doctor", "Entrepreneur", "Artist", "Researcher", "Other"])
financial_status = st.sidebar.selectbox("Financial Stability / நிதி நிலை", ["Low", "Medium", "High"])
dependents = st.sidebar.number_input("Number of Dependents / சார்ந்திருப்போர் எண்ணிக்கை", 0, 10, 0)
sponsorship = st.sidebar.selectbox("Do you have a Sponsor? / உங்களுக்கு நிதி ஆதரவு உள்ளதா?", ["Yes", "No"])
previous_visas = st.sidebar.selectbox("Have you had previous UK visas? / முன்பு யுகே விசா பெற்றுள்ளீர்களா?", ["Yes", "No"])

st.markdown("### ✈️ Applicant Information Form / விண்ணப்பதாரர் தகவல் படிவம்")
st.write("Please fill the details carefully to get the most suitable visa recommendation. / மிகச் சிறந்த விசா பரிந்துரையை பெற விவரங்களை கவனமாக நிரப்பவும்.")

# --------------------------
# Encode Inputs
# --------------------------
def encode_inputs():
    return pd.DataFrame([{
        "Applicant Age": age,
        "Education Level": {"High School": 1, "Diploma": 2, "Bachelor's": 3, "Master's": 4, "PhD": 5}[education],
        "Occupation": {"Student": 1, "Engineer": 2, "Doctor": 3, "Entrepreneur": 4, "Artist": 5, "Researcher": 6, "Other": 7}[occupation],
        "Financial Status": {"Low": 1, "Medium": 2, "High": 3}[financial_status],
        "Dependents": dependents,
        "Sponsorship Status": 1 if sponsorship == "Yes" else 0,
        "Previous UK Visas": 1 if previous_visas == "Yes" else 0
    }])

# --------------------------
# Visa Step Guides (English + Tamil)
# --------------------------
visa_steps = {
    "Work & Employment Visa": [
        "✅ Check eligibility and confirm your job offer is valid.",
        "📄 Prepare documents: passport, employment letter, certificates, police clearance.",
        "💻 Apply online and complete medical checks.",
        "💰 Pay visa fees and attend biometrics appointment."
    ],
    "Student & Academic Visa": [
        "🎓 Obtain admission confirmation (CAS/I-20).",
        "📑 Prepare financial documents and transcripts.",
        "💻 Apply online and attend interview if required."
    ],
    "Family & Dependent Visa": [
        "👨‍👩‍👧 Confirm relationship and sponsor eligibility.",
        "📄 Submit certificates, sponsorship proof, and relationship evidence."
    ],
    "Visitor & Tourism Visa": [
        "🌍 Decide travel purpose and prepare itinerary.",
        "💻 Apply via official portal and pay fees."
    ],
    "Permanent Residency & Settlement Visa": [
        "🏡 Check eligibility for PR/settlement route.",
        "📋 Submit work experience, language proof, and funds documents."
    ],
    "Special Category Visa": [
        "🎭 Submit proof of special eligibility (artist, diplomat, sports, etc.)."
    ]
}

tamil_steps = {
    "Work & Employment Visa": [
        "✅ தகுதி மற்றும் வேலை வாய்ப்பை உறுதிசெய்யவும்.",
        "📄 பாஸ்போர்ட், வேலை கடிதம், சான்றிதழ்கள், போலீஸ் சர்டிபிகேட் தயாரிக்கவும்.",
        "💻 இணையதளம் வழியாக விண்ணப்பிக்கவும் மற்றும் மருத்துவ பரிசோதனை செய்யவும்.",
        "💰 கட்டணத்தை செலுத்தி பயோமெட்ரிக் பதிவு செய்யவும்."
    ],
    "Student & Academic Visa": [
        "🎓 கல்வி நிறுவனத்தில் சேர்க்கை உறுதிப்படுத்தவும்.",
        "📑 நிதி ஆவணங்கள் மற்றும் மதிப்பெண் பட்டியல்கள் தயாரிக்கவும்.",
        "💻 மாணவர் விசா இணையதளத்தின் மூலம் விண்ணப்பிக்கவும்."
    ],
    "Family & Dependent Visa": [
        "👨‍👩‍👧 உறவு மற்றும் ஸ்பான்சர் தகுதியை உறுதிப்படுத்தவும்.",
        "📄 சான்றிதழ்கள் மற்றும் ஆதார ஆவணங்களை சமர்ப்பிக்கவும்."
    ],
    "Visitor & Tourism Visa": [
        "🌍 பயண நோக்கத்தை தீர்மானிக்கவும் மற்றும் பயண திட்டம் தயாரிக்கவும்.",
        "💻 அதிகாரப்பூர்வ தளத்தின் மூலம் விண்ணப்பித்து கட்டணம் செலுத்தவும்."
    ],
    "Permanent Residency & Settlement Visa": [
        "🏡 நிரந்தர குடியேற்ற தகுதியை சரிபார்க்கவும்.",
        "📋 வேலை அனுபவம் மற்றும் நிதி ஆதாரங்களை சமர்ப்பிக்கவும்."
    ],
    "Special Category Visa": [
        "🎭 சிறப்பு தகுதி சான்றுகள் (கலைஞர், தூதுவர், விளையாட்டு வீரர் போன்றவை) சமர்ப்பிக்கவும்."
    ]
}

# --------------------------
# Prediction Button
# --------------------------
if st.button("🔍 Recommend My Visa / எனது விசாவை பரிந்துரைக்கவும்"):
    input_data = encode_inputs()
    prediction = model.predict(input_data)
    visa_type = label_encoder.inverse_transform(prediction)[0]

    st.markdown(f"<div class='visa'>🎯 Recommended Visa Type / பரிந்துரைக்கப்பட்ட விசா வகை: <b>{visa_type}</b></div>", unsafe_allow_html=True)

    st.subheader("📘 Step-by-Step Procedure (English):")
    for step in visa_steps.get(visa_type, []):
        st.write(step)

    if visa_type in tamil_steps:
        st.subheader("📙 விசா விண்ணப்பிக்கும் படிகள் (தமிழ்):")
        for step in tamil_steps[visa_type]:
            st.write(step)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("✨ *Developed by Team Purity & Destiny (AI Project – Visa Recommendation System)* | 💡 *புரிட்டி & டெஸ்டினி குழுவால் உருவாக்கப்பட்டது*")
