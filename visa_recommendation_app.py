import streamlit as st
import pandas as pd
import joblib
import zipfile
import io
from sklearn.preprocessing import StandardScaler
import numpy as np

# Custom CSS for Purity and Destiny theme
st.markdown("""
    <style>
    /* General styling for purity (clean, white, serene) */
    .main {
        background-color: #F5F7FA;
        font-family: 'Roboto', sans-serif;
        color: #2C3E50;
    }
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                         url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3 {
        color: #1A5276;
        font-weight: 300;
        text-align: center;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .stTextInput>div>input {
        border: 1px solid #3498DB;
        border-radius: 5px;
        background-color: #FFFFFF;
    }
    .stRadio>label {
        color: #2C3E50;
        font-size: 16px;
    }
    .stFileUploader>label {
        color: #2C3E50;
        font-size: 16px;
    }
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained Random Forest model and label encoder
model = joblib.load("best_visa_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define globally applicable visa application procedures
visa_procedures = {
    'Visitor': [
        "1. Check if a visa is required for your travel purpose (e.g., tourism, business) and destination country.",
        "2. Complete the visa application form (online or paper-based) as required by the destination country.",
        "3. Pay the applicable visa processing fee, if required.",
        "4. Schedule an appointment or submit your application at the relevant embassy, consulate, or visa application center.",
        "5. Gather required documents (valid passport, photographs, travel itinerary, proof of funds, accommodation details, etc.).",
        "6. Attend an interview or biometric appointment, if required by the country.",
        "7. Await visa processing and collect your visa or receive approval notification."
    ],
    'PR': [
        "1. Determine eligibility for permanent residency (e.g., through family ties, employment, investment, or humanitarian grounds).",
        "2. Identify a sponsor or meet self-sponsored criteria, if applicable, as per the country's immigration policy.",
        "3. Submit an application for permanent residency through the immigration authority or embassy.",
        "4. Provide supporting documents (identification, proof of relationship, employment history, financial statements, etc.).",
        "5. Complete any required medical examinations or background checks.",
        "6. Attend an interview or provide additional information, if requested.",
        "7. Receive permanent residency approval and comply with residency conditions."
    ],
    'Student': [
        "1. Secure admission to an accredited educational institution and obtain an acceptance letter.",
        "2. Pay any required fees for visa processing or student registration (e.g., SEVIS-like fees in some countries).",
        "3. Complete the student visa application form for the destination country.",
        "4. Gather required documents (passport, acceptance letter, proof of financial support, academic records, etc.).",
        "5. Schedule and attend a visa interview or biometric appointment, if required.",
        "6. Pay the visa application fee, if applicable.",
        "7. Receive your student visa and prepare for enrollment."
    ],
    'Work': [
        "1. Secure a job offer or employment contract from an employer in the destination country.",
        "2. Verify if the employer needs to obtain a work permit or labor certification on your behalf.",
        "3. Complete the work visa application form as required by the country's immigration authority.",
        "4. Gather required documents (passport, job offer letter, qualifications, professional certifications, etc.).",
        "5. Pay the visa application fee, if applicable.",
        "6. Attend a visa interview or provide biometrics, if required.",
        "7. Receive your work visa and comply with employment regulations."
    ],
    'Family': [
        "1. Confirm eligibility for family reunification based on relationship to a resident or citizen of the destination country.",
        "2. Have a sponsor (family member) submit a sponsorship or reunification application, if required.",
        "3. Complete the family visa application form for the destination country.",
        "4. Provide supporting documents (proof of relationship, sponsor’s financial documents, marriage/birth certificates, etc.).",
        "5. Attend a visa interview or medical examination, if required.",
        "6. Pay any applicable visa or processing fees.",
        "7. Receive your family visa and join your family in the destination country."
    ],
    'Special': [
        "1. Identify eligibility for a special visa (e.g., for extraordinary abilities, humanitarian reasons, or specific programs).",
        "2. Secure sponsorship or nomination, if required by the destination country’s immigration policy.",
        "3. Complete the relevant visa application form for special categories.",
        "4. Provide evidence of eligibility (awards, professional achievements, humanitarian need, etc.).",
        "5. Submit supporting documents (passport, letters of recommendation, proof of achievements, etc.).",
        "6. Attend an interview or provide additional verification, if requested.",
        "7. Receive your special visa and comply with its conditions."
    ]
}

# Load the original dataset to get feature scaling parameters
original_data = pd.read_csv("preprocessed_dataset.csv")
X_original = original_data.drop(columns=["visa_category (Label)"])
scaler = StandardScaler().fit(X_original)

# Streamlit app layout
st.title("Global Visa Recommendation System")
st.markdown("<h3>Embark on Your Journey to a New Future</h3>", unsafe_allow_html=True)
st.write("Input your details or upload a ZIP file to discover the right visa for your global journey.")

# Option to choose between single input or batch upload
option = st.radio("Select input method:", ("Single Applicant", "Batch Upload (ZIP)"), horizontal=True)

if option == "Single Applicant":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Your Details")
    
    # Input fields for features (based on preprocessed_dataset.csv)
    purpose_of_travel = st.number_input("Purpose of Travel (normalized)", value=0.0, step=0.1)
    age = st.number_input("Age (normalized)", value=0.0, step=0.1)
    current_residence = st.number_input("Current Residence (normalized)", value=0.0, step=0.1)
    work_experience_years = st.number_input("Work Experience Years (normalized)", value=0.0, step=0.1)
    nationality = st.number_input("Nationality (normalized)", value=0.0, step=0.1)
    marital_status = st.number_input("Marital Status (normalized)", value=0.0, step=0.1)
    dependents = st.number_input("Dependents (normalized)", value=0.0, step=0.1)
    education_level = st.number_input("Education Level (normalized)", value=0.0, step=0.1)
    occupation = st.number_input("Occupation (normalized)", value=0.0, step=0.1)
    document_completeness_score = st.number_input("Document Completeness Score (normalized)", value=0.0, step=0.1)
    financial_status = st.number_input("Financial Status (normalized)", value=0.0, step=0.1)
    intended_duration = st.number_input("Intended Duration (normalized)", value=0.0, step=0.1)
    sponsorship_status = st.number_input("Sponsorship Status (normalized)", value=0.0, step=0.1)
    previous_visa_rejections = st.number_input("Previous Visa Rejections (normalized)", value=0.0, step=0.1)
    language_proficiency = st.number_input("Language Proficiency (normalized)", value=0.0, step=0.1)

    # Collect inputs into a DataFrame
    input_data = pd.DataFrame({
        "purpose_of_travel": [purpose_of_travel],
        "age": [age],
        "current_residence": [current_residence],
        "work_experience_years": [work_experience_years],
        "nationality": [nationality],
        "marital_status": [marital_status],
        "dependents": [dependents],
        "education_level": [education_level],
        "occupation": [occupation],
        "document_completeness_score": [document_completeness_score],
        "financial_status": [financial_status],
        "intended_duration": [intended_duration],
        "sponsorship_status": [sponsorship_status],
        "previous_visa_rejections": [previous_visa_rejections],
        "language_proficiency": [language_proficiency]
    })

    if st.button("Discover Your Visa Path"):
        # Apply the same scaling as the training data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        visa_type = label_encoder.inverse_transform(prediction)[0]
        
        # Display results
        st.subheader("Your Visa Recommendation")
        st.write(f"**Recommended Visa Type**: {visa_type}")
        st.write("**Steps to Apply (General Guidelines)**:")
        for step in visa_procedures.get(visa_type, ["No procedures available."]):
            st.write(f"• {step}")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload ZIP File for Batch Predictions")
    uploaded_file = st.file_uploader("Choose a ZIP file containing a CSV", type=["zip"])

    if uploaded_file is not None:
        # Read the ZIP file
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            # Assume the ZIP contains a single CSV file
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                with zip_ref.open(csv_files[0]) as csv_file:
                    batch_data = pd.read_csv(csv_file)
                    
                    # Verify the CSV has the correct columns
                    expected_columns = X_original.columns
                    if set(expected_columns).issubset(batch_data.columns):
                        # Drop any extra columns and keep only the required features
                        batch_data = batch_data[expected_columns]
                        
                        # Apply scaling
                        batch_data_scaled = scaler.transform(batch_data)
                        
                        # Make predictions
                        predictions = model.predict(batch_data_scaled)
                        visa_types = label_encoder.inverse_transform(predictions)
                        
                        # Add predictions to the DataFrame
                        batch_data['Predicted_Visa_Type'] = visa_types
                        
                        # Display results
                        st.subheader("Batch Prediction Results")
                        st.dataframe(batch_data)
                        
                        # Provide download link for results
                        csv_buffer = io.StringIO()
                        batch_data.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv_buffer.getvalue(),
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Display procedures for each unique visa type
                        unique_visa_types = set(visa_types)
                        for visa_type in unique_visa_types:
                            st.subheader(f"Application Steps for {visa_type} (General Guidelines)")
                            for step in visa_procedures.get(visa_type, ["No procedures available."]):
                                st.write(f"• {step}")
                    else:
                        st.error("CSV file does not contain the required columns.")
            else:
                st.error("No CSV file found in the ZIP archive.")
    st.markdown("</div>", unsafe_allow_html=True)