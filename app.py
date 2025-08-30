import streamlit as st
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

# Azure Form Recognizer (OCR) Setup
form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
form_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint,
    credential=AzureKeyCredential(form_recognizer_key)
)

# Azure OpenAI Setup
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=openai_key,
    api_version="2025-01-01-preview",
    azure_endpoint=openai_endpoint
)

# --------------------------
# Streamlit UI
# --------------------------
st.title("AI Medical Report Analyzer (Local Prototype)")
st.write("**Disclaimer:** This tool is for educational/testing purposes only. It is **not** a medical diagnosis tool.")

# Step 1: User Inputs
uploaded_file = st.file_uploader("Upload your medical report (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Select gender", ["Male", "Female", "Other"])
notes = st.text_area("Any other relevant info (optional)")

# --------------------------
# Step 2 & 3: Analyze Report and Generate Questions
# --------------------------
if uploaded_file and st.button("Analyze Report", key="analyze_report"):
    with st.spinner("Extracting text from report..."):
        poller = form_client.begin_analyze_document("prebuilt-document", document=uploaded_file)
        result = poller.result()

        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"

        st.session_state["extracted_text"] = extracted_text

    st.subheader("Extracted Text from Report")
    st.write(st.session_state["extracted_text"])

    with st.spinner("Generating follow-up questions..."):
        question_prompt = f"""
        You are a medical assistant AI. Based on this report and patient details:
        Age: {age}, Gender: {gender}, Notes: {notes}
        Report text:
        {st.session_state['extracted_text']}
        Suggest 3-4 follow-up questions to ask the patient to better understand possible conditions.
        Return them as a numbered list.
        """
        question_response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": question_prompt}
            ]
        )
        questions_text = question_response.choices[0].message.content.strip()

        # Split into list
        questions_list = [q.strip() for q in questions_text.split("\n") if q.strip()]
        st.session_state["follow_up_questions"] = questions_list
        st.session_state["answers"] = [""] * len(questions_list)

# --------------------------
# Step 4: Show Questions & Collect Answers
# --------------------------
if "follow_up_questions" in st.session_state:
    st.subheader("Follow-Up Questions")
    for i, q in enumerate(st.session_state["follow_up_questions"]):
        st.session_state["answers"][i] = st.text_input(
            f"{q}", 
            value=st.session_state["answers"][i], 
            key=f"answer_{i}"
        )

    if st.button("Submit Answers", key="submit_answers"):
        with st.spinner("Analyzing possible conditions..."):
            condition_prompt = f"""
            You are a medical assistant AI. Based on:
            - Patient Age: {age}
            - Gender: {gender}
            - Notes: {notes}
            - Report: {st.session_state['extracted_text']}
            - Patient Answers: {st.session_state['answers']}

            Suggest possible health conditions (3 max) and a way forward.
            Include a disclaimer that this is not a diagnosis.
            """
            condition_response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": condition_prompt}
                ]
            )
            st.session_state["possible_conditions"] = condition_response.choices[0].message.content

# --------------------------
# Step 5: Show Results if Available
# --------------------------
if "possible_conditions" in st.session_state:
    st.subheader("Possible Conditions & Advice")
    st.write(st.session_state["possible_conditions"])
