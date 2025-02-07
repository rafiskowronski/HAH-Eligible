import streamlit as st
import pandas as pd
import requests
import openai  # For NLP-based text analysis

# -------------------- CONFIGURATION -------------------- #
# Replace with actual Epic FHIR API URL
FHIR_API_URL = "https://epic.fhir.server/patient-data"

# OpenAI API Key (Replace with actual API key)
OPENAI_API_KEY = "your_openai_api_key"
openai.api_key = OPENAI_API_KEY

# Function to fetch patient structured data from Epic
def fetch_patient_data(patient_id):
    """ Simulates fetching structured patient data from Epic EMR (FHIR API) """
    response = requests.get(f"{FHIR_API_URL}/{patient_id}")
    if response.status_code == 200:
        return response.json()  # Actual API response
    else:
        # Simulated patient data
        return {
            "patient_id": patient_id,
            "age": 75,
            "diagnosis": "CHF",
            "ed_visits": 2,
            "icu_stay": "No",
            "o2_sat": 92,
            "cognitive_status": "None",
            "adls_independent": "Yes",
            "caregiver_available": "Yes",
            "telehealth_history": "Yes",
            "internet_access": "Yes",
        }

# Function to fetch free-text clinical notes from Epic
def fetch_clinical_notes(patient_id):
    """ Simulates fetching free-text clinical notes from Epic """
    return """
    The patient is medically stable with no acute distress. Home environment is safe, and a caregiver is present. 
    History of CHF but well managed with current medications. No recent falls. 
    Patient has used telehealth services in the past and reports confidence in managing care remotely.
    """

# Function to analyze clinical notes using NLP
def analyze_clinical_notes(notes):
    """ Uses OpenAI NLP model to analyze clinical notes for Hospital at Home suitability """
    prompt = f"Analyze the following clinical notes and determine if the patient is eligible for Hospital at Home. \
    Consider stability, caregiver support, technology readiness, and risk factors. Give a concise eligibility assessment:\n\n{notes}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

# Function to determine AI-based eligibility
def determine_priority(patient_data, nlp_analysis):
    """ AI-driven patient prioritization based on structured data + NLP insights """
    
    if patient_data["icu_stay"] == "Yes":
        return "❌ Low Priority (Likely Ineligible)", 0
    if patient_data["o2_sat"] < 90:
        return "⚠️ Medium Priority (Needs Review)", 1
    if "high fall risk" in nlp_analysis.lower() or "requires iv vasopressors" in nlp_analysis.lower():
        return "❌ Low Priority (Likely Ineligible)", 0
    if "caregiver present" in nlp_analysis.lower() and "managing care remotely" in nlp_analysis.lower():
        return "✅ High Priority (Likely Eligible)", 2
    
    return "⚠️ Medium Priority (Needs Review)", 1

# -------------------- STREAMLIT UI -------------------- #
st.title("🏥 AI-Powered Hospital at Home Ranking System")

# Input list of patient IDs
patient_ids = st.text_area("Enter Patient IDs (comma-separated):", "PT-001, PT-002, PT-003")

if st.button("Fetch & Rank Patients"):
    patient_list = patient_ids.split(",")

    ranking_results = []
    
    for patient_id in patient_list:
        patient_id = patient_id.strip()
        patient_data = fetch_patient_data(patient_id)
        clinical_notes = fetch_clinical_notes(patient_id)
        nlp_analysis = analyze_clinical_notes(clinical_notes)
        priority, priority_score = determine_priority(patient_data, nlp_analysis)

        ranking_results.append({
            "Patient ID": patient_data["patient_id"],
            "Age": patient_data["age"],
            "Diagnosis": patient_data["diagnosis"],
            "ED Visits (6M)": patient_data["ed_visits"],
            "O2 Sat (%)": patient_data["o2_sat"],
            "Caregiver Available": patient_data["caregiver_available"],
            "Telehealth History": patient_data["telehealth_history"],
            "Internet Access": patient_data["internet_access"],
            "AI-Predicted Priority": priority,
            "Priority Score": priority_score  # Used for sorting
        })

    # Convert to DataFrame and sort by AI priority score
    ranking_df = pd.DataFrame(ranking_results)
    ranking_df = ranking_df.sort_values(by="Priority Score", ascending=False)

    # Display ranked results
    st.write("### Ranked Patients for Hospital at Home")
    st.dataframe(ranking_df)