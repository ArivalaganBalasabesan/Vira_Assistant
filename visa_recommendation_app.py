# app.py — VIRA • Sri Lanka Visa Assistant (bright Sri Lanka theme)
# Notes:
# - Replaced deprecated st.experimental_rerun() with st.rerun()
# - Bright, friendly UI tuned to Sri Lankan color palette
# - Minor UX polish (placeholders, consistent buttons, better spacing)
# - Safe fallbacks for encoders and feature alignment

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="VIRA • Sri Lanka Visa Assistant",
    page_icon="🇱🇰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Global Styles — bright theme
# -----------------------------
# Colors inspired by the Sri Lankan flag (saffron, green, maroon, gold)
PRIMARY = "#8D1B3D"   # maroon
ACCENT  = "#FF9933"   # saffron
SUCCESS = "#138808"   # green
GOLD    = "#FFD54F"   # golden yellow
LIGHT_BG = "#FAFAFA"

st.markdown(
    f"""
    <style>
      html, body, [class*="css"]  {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; }}
      .top-banner {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        color: white; padding: 14px 18px; border-radius: 12px; margin: 10px 0 20px 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
      }}
      .subtitle {{ text-align:center; color:#333; margin-top: -8px; }}
      .chatbox {{ max-width: 920px; margin: 14px auto; padding: 16px; background: rgba(255,255,255,0.98); border-radius: 14px; box-shadow: 0 8px 26px rgba(0,0,0,0.08); }}
      .user {{ background: #E8F5E9; padding: 10px 14px; border-radius: 18px; text-align: right; margin: 8px; display: block; border: 1px solid #C8E6C9; }}
      .bot  {{ background: #F3E5F5; padding: 10px 14px; border-radius: 18px; text-align: left;  margin: 8px; display: block; border: 1px solid #E1BEE7; }}
      .badge {{ font-size: 0.9em; padding:6px 10px; border-radius: 8px; background: {SUCCESS}; color:white; }}
      .question-card {{ background: {LIGHT_BG}; border: 1px solid #eee; padding: 16px; border-radius: 12px; }}
      .hint {{ font-size: 0.9em; color: #555; }}

      /* Buttons */
      .stButton>button {{
        background: {ACCENT}; color: white; border: none; padding: 0.6rem 1rem; border-radius: 10px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
      }}
      .stButton>button:hover {{ filter: brightness(1.05); }}

      /* Number input "stepper" tweaks for consistency */
      div[data-baseweb="input"] input {{ font-size: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="top-banner"><h2 style="margin:0">🇱🇰 VIRA — Sri Lanka Visa Recommendation Assistant</h2></div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>Bright, friendly guidance for visas — designed for Sri Lankans at home and abroad.</p>",
    unsafe_allow_html=True,
)

# -----------------------------
# Load model + encoders (cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts["model"] = joblib.load("visa_recommendation_model.pkl")
    except Exception as e:
        artifacts["model"] = None
        artifacts["model_error"] = str(e)
    # optional saved encoders
    try:
        artifacts["label_encoder"] = joblib.load("label_encoder.pkl")
    except:
        artifacts["label_encoder"] = None
    try:
        artifacts["feature_encoders"] = joblib.load("feature_encoders.pkl")
    except:
        artifacts["feature_encoders"] = None
    return artifacts

art = load_artifacts()
model = art.get("model")
label_encoder = art.get("label_encoder")
feature_encoders = art.get("feature_encoders")

if model is None:
    st.error("❌ Could not load model file 'visa_recommendation_model.pkl'. Make sure it is in the same folder.")
    st.stop()

# -----------------------------
# Encoding helper
# -----------------------------

def encode_with_feature_encoders(df_in: pd.DataFrame):
    """Use saved LabelEncoders if present. Expand unseen classes safely and
    return a numeric DF aligned as best as possible to model features."""
    df = df_in.copy()
    if feature_encoders is None:
        return None
    for col, enc in feature_encoders.items():
        if col in df.columns:
            val = df.loc[0, col]
            if str(val) not in enc.classes_.astype(str):
                enc.classes_ = np.append(enc.classes_, str(val))
            df[col] = enc.transform([str(val)])[0]
    return df

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "👋 Ayubowan! I'm VIRA — your Sri Lanka Visa Assistant. I'll ask a few quick questions to recommend the best visa type."}
    ]
    st.session_state.step = 0
    st.session_state.answers = {}

# Question flow
questions = [
    ("purpose_of_travel", "What is your main purpose of travel?", ["Work", "Study", "Tourism", "Family", "Settlement", "Other"], "choice"),
    ("age", "How old are you?", None, "number"),
    ("current_residence", "Where do you currently reside?", None, "text"),
    ("work_experience_years", "Years of work experience?", None, "number"),
    ("education_level", "Highest education", ["High School", "Diploma", "Bachelor", "Master", "PhD"], "choice"),
    ("occupation", "Your occupation / job title?", None, "text"),
    ("financial_status", "Financial situation (choose closest):", ["Low", "Moderate", "Strong"], "choice"),
    ("sponsorship_status", "Do you have a sponsor?", ["Yes","No"], "choice"),
    ("dependents", "How many dependents do you have?", None, "number"),
    ("previous_visa_rejections", "Any previous visa rejections? (number)", None, "number"),
    ("language_proficiency", "Language proficiency", ["Basic","Intermediate","Fluent"], "choice"),
]

# Utilities

def show_chat():
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        css = "bot" if m["role"] == "bot" else "user"
        st.markdown(f"<div class='{css}'>{m['text']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

show_chat()

# -----------------------------
# Question UI
# -----------------------------
if st.session_state.step < len(questions):
    key, prompt, opts, qtype = questions[st.session_state.step]

    # Ask once per step
    if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["text"] != prompt:
        st.session_state.messages.append({"role": "bot", "text": prompt})
        show_chat()

    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    if qtype == "choice":
        choice = st.selectbox("", opts, key=f"q_{key}", index=0, placeholder="Select an option…")
        if st.button("Next ▶"):
            st.session_state.answers[key] = choice
            st.session_state.messages.append({"role": "user", "text": str(choice)})
            st.session_state.step += 1
            st.rerun()

    elif qtype == "number":
        default = 0 if key != "age" else 25
        num = st.number_input("", min_value=0, max_value=120, value=default, step=1, key=f"q_{key}")
        if st.button("Next ▶"):
            st.session_state.answers[key] = int(num)
            st.session_state.messages.append({"role": "user", "text": str(int(num))})
            st.session_state.step += 1
            st.rerun()

    else:  # text
        placeholder = "Type your answer here…"
        txt = st.text_input("", key=f"q_{key}", placeholder=placeholder)
        if st.button("Next ▶"):
            st.session_state.answers[key] = txt.strip()
            st.session_state.messages.append({"role": "user", "text": txt.strip()})
            st.session_state.step += 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # All answers collected → predict
    st.session_state.messages.append({"role": "bot", "text": "🔎 Analyzing your profile… Please wait a moment."})
    show_chat()
    time.sleep(0.8)

    user_df = pd.DataFrame([st.session_state.answers])

    # Try saved feature_encoders first
    encoded = None
    try:
        if feature_encoders:
            encoded = encode_with_feature_encoders(user_df)
    except Exception:
        encoded = None

    if encoded is None:
        # Fallback: one-hot and align
        encoded = pd.get_dummies(user_df.astype(str))
        model_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if model_cols:
            for c in model_cols:
                if c not in encoded.columns:
                    encoded[c] = 0
            encoded = encoded[model_cols]
    else:
        model_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if model_cols:
            for c in model_cols:
                if c not in encoded.columns:
                    encoded[c] = 0
            encoded = encoded[model_cols]

    if hasattr(model, "feature_names_in_"):
        encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        # --- Predict with robust label resolution ---
        proba, best_idx = None, None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(encoded)
            best_idx = int(np.argmax(probs, axis=1)[0])
            proba = float(probs[0, best_idx])

        pred = model.predict(encoded)[0]

        pred_label = None
        # 1) If a saved label encoder exists, prefer it
        if label_encoder is not None:
            try:
                pred_label = label_encoder.inverse_transform([pred])[0]
            except Exception:
                try:
                    pred_label = label_encoder.inverse_transform([int(pred)])[0]
                except Exception:
                    pred_label = None

        # 2) Else, try model.classes_ (works when model trained on string labels)
        if pred_label is None and hasattr(model, "classes_"):
            classes = list(model.classes_)
            if any(isinstance(c, str) for c in classes):
                if best_idx is not None and 0 <= best_idx < len(classes):
                    pred_label = classes[best_idx]
                else:
                    try:
                        idx = classes.index(pred)
                        pred_label = classes[idx]
                    except ValueError:
                        pass

        # 3) Final fallback: numeric code → friendly name mapping
        if pred_label is None:
            code_to_name = {
                0: "Work & Employment Visa",
                1: "Student & Academic Visa",
                2: "Family & Dependent Visa",
                3: "Visitor & Tourism Visa",
                4: "Permanent Residency & Settlement Visa",
                5: "Special Category Visa",
            }
            try:
                pred_label = code_to_name.get(int(pred), str(pred))
            except Exception:
                pred_label = str(pred)

    except Exception as e:
        st.session_state.messages.append({"role": "bot", "text": f"⚠️ Prediction failed: {e}"})
        show_chat()
        st.stop()

    # Human-friendly output
    st.session_state.messages.append({
        "role": "bot",
        "text": f"🏆 Recommendation: <span class='badge'><strong>{pred_label}</strong></span>" + (f" (confidence {proba:.2f})" if proba is not None else "")
    })
    show_chat()

    def get_visa_steps(visa_type: str):
        steps = {
            "Work & Employment Visa": [
                "Check eligibility and obtain a validated job offer.",
                "Collect passport, employment letter, certificates, police & medical checks.",
                "Apply online at the embassy or immigration portal.",
                "Pay fees, book biometrics, attend interview if requested.",
                "Wait for decision and collect your passport.",
            ],
            "Student & Academic Visa": [
                "Secure admission and obtain offer letter.",
                "Prepare transcripts, proof of funds and health checks.",
                "Apply through official channels and attend biometrics.",
                "Receive visa and arrange travel & accommodation.",
            ],
            "Family & Dependent Visa": [
                "Confirm sponsor's legal status and prepare relationship proofs.",
                "Submit dependent visa application with sponsor documents.",
                "Attend biometrics if required and wait for decision.",
            ],
            "Visitor & Tourism Visa": [
                "Prepare travel itinerary, hotel bookings & proof of funds.",
                "Apply online, attend biometrics if required.",
                "Receive visa and travel with travel insurance.",
            ],
            "Permanent Residency & Settlement Visa": [
                "Check PR eligibility and collect long-term records.",
                "Submit PR application and complete checks.",
                "Wait for approval and collect PR documentation.",
            ],
            "Special Category Visa": [
                "Identify subcategory and secure invitation/endorsement.",
                "Submit required documents and follow special instructions.",
                "Attend processes required for the category and receive approval.",
            ],
        }
        return steps.get(visa_type, ["⚠️ Visa procedure not available for this category."])

    st.session_state.messages.append({"role": "bot", "text": "📋 Here are the recommended next steps for your application:"})
    show_chat()
    for s in get_visa_steps(str(pred_label)):
        st.session_state.messages.append({"role": "bot", "text": s})
        show_chat()
        time.sleep(0.45)

    st.session_state.messages.append({"role": "bot", "text": "✅ Good luck! You can export this conversation or try another profile."})
    show_chat()

# -----------------------------
# Footer actions
# -----------------------------
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔁 Reset Chat"):
        st.session_state.messages = [{"role": "bot", "text": "👋 Ayubowan! I'm VIRA — your Sri Lanka Visa Assistant. I'll ask a few quick questions to recommend the best visa type."}]
        st.session_state.step = 0
        st.session_state.answers = {}
        st.rerun()
with col_b:
    transcript = "\n".join([f"{m['role']}: {m['text']}" for m in st.session_state.messages])
    st.download_button("📥 Download Transcript (.txt)", transcript, file_name="vira_chat_transcript.txt")
