import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Buat hubungin dengan api.py
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Placement Prediction (Decoupled)", layout="wide")
st.title("Student Placement & Salary Prediction")
st.markdown("### *Decoupled Architecture: Streamlit Frontend + FastAPI Backend*")
st.markdown("---")

left_col, right_col = st.columns([1.2, 1], gap="medium")

with left_col:
    st.markdown("### Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        hsc = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        degree = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        entrance = st.number_input("Entrance Exam Score", min_value=0, max_value=100, value=60)
        tech_skill = st.number_input("Technical Skill Score", min_value=0, max_value=100, value=70)
        soft_skill = st.number_input("Soft Skill Score", min_value=0, max_value=100, value=70)
    
    with col2:
        internship = st.number_input("Internship Count", min_value=0, max_value=10, value=1)
        live_projects = st.number_input("Live Projects", min_value=0, max_value=20, value=2)
        work_exp = st.number_input("Work Experience (months)", min_value=0, max_value=60, value=0)
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=0)
        attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        backlogs = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])


payload = {
    "gender": gender,
    "ssc_percentage": ssc,
    "hsc_percentage": hsc,
    "degree_percentage": degree,
    "cgpa": cgpa,
    "entrance_exam_score": entrance,
    "technical_skill_score": tech_skill,
    "soft_skill_score": soft_skill,
    "internship_count": internship,
    "live_projects": live_projects,
    "work_experience_months": work_exp,
    "certifications": certifications,
    "attendance_percentage": attendance,
    "backlogs": backlogs,
    "extracurricular_activities": extracurricular
}

with right_col:
    st.markdown("### Prediction via FastAPI")
    
    if st.button("Predict Placement & Salary", use_container_width=True):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                placement = result["placement_status"]
                salary = result["predicted_salary_lpa"]
                
                st.markdown("---")
                st.subheader("Placement Result")
                if placement == 1:
                    st.success("Student is likely to be **PLACED**")
                    st.subheader("Predicted Salary Package")
                    st.info(f"**{salary:.2f} LPA**")
                else:
                    st.error("Student is likely **NOT to be placed**")
                    st.subheader("Predicted Salary Package")
                    st.info(f"**{salary:.2f} LPA** (Not placed)")
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server. Make sure it's running at " + API_URL)
    
    st.markdown("---")
    st.markdown("### Data Visualization")
    
    scores_df = pd.DataFrame({
        'Metric': ['SSC %', 'HSC %', 'Degree %', 'CGPA (scaled)',
                   'Technical Skill', 'Soft Skill', 'Entrance Exam', 'Attendance'],
        'Value': [ssc, hsc, degree, cgpa*10, tech_skill, soft_skill, entrance, attendance]
    })
    
    fig = px.bar(scores_df, x='Metric', y='Value',
                 title='Student Performance Metrics',
                 labels={'Value': 'Score (%)'},
                 color='Value', color_continuous_scale='Blues',
                 text='Value')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(showlegend=False, height=450,
                      xaxis_tickangle=-45,
                      yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Frontend: Streamlit | Backend: FastAPI (Model prediction via API)")