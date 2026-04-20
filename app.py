import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

clf = joblib.load('artifacts/xgb_classifier.pkl')
reg = joblib.load('artifacts/linear_regression.pkl')

st.set_page_config(page_title="Placement Prediction", layout="wide")
st.title("Student Placement & Salary Prediction")
st.markdown("---")

left_col, right_col = st.columns([1.2, 1], gap="medium")

# Input Data 
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

academic_score = (ssc + hsc + degree + cgpa * 10) / 4

input_data = pd.DataFrame([{
    'gender': gender,
    'ssc_percentage': ssc,
    'hsc_percentage': hsc,
    'degree_percentage': degree,
    'cgpa': cgpa,
    'entrance_exam_score': entrance,
    'technical_skill_score': tech_skill,
    'soft_skill_score': soft_skill,
    'internship_count': internship,
    'live_projects': live_projects,
    'work_experience_months': work_exp,
    'certifications': certifications,
    'attendance_percentage': attendance,
    'backlogs': backlogs,
    'extracurricular_activities': extracurricular,
    'academic_score': academic_score
}])

#Prediction + Visualization
with right_col:
    st.markdown("### Prediction")
    
    # Button to trigger prediction
    if st.button("Predict Placement & Salary", use_container_width=True):
        placement_pred = clf.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Placement Result")
        if placement_pred == 1:
            st.success("Student is likely to be **PLACED**")
            salary_pred = reg.predict(input_data)[0]
            st.subheader("Predicted Salary Package")
            st.info(f"**{salary_pred:.2f} LPA**")
        else:
            st.error("Student is likely **NOT to be placed**")
            salary_pred = 0.0
            st.subheader("Predicted Salary Package")
            st.info(f"**{salary_pred:.2f} LPA** (Not placed)")
  
    
    st.markdown("---")
    st.markdown("### Data Visualization")
    
    # Visualization function: show key metrics as bar chart
    # Create a dataframe of important scores
    scores_df = pd.DataFrame({
        'Metric': ['SSC %', 'HSC %', 'Degree %', 'CGPA (scaled)', 'Academic Score', 
                   'Technical Skill', 'Soft Skill', 'Entrance Exam', 'Attendance'],
        'Value': [ssc, hsc, degree, cgpa*10, academic_score, tech_skill, soft_skill, entrance, attendance],
        'Max Score': [100, 100, 100, 100, 100, 100, 100, 100, 100]
    })
    
    # Bar chart using Plotly
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
    
    
        
# python -m streamlit run app.py