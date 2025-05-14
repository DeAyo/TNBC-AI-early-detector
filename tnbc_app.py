# tnbc_app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='TNBC Risk App', layout='centered')
st.title('🧬 TNBC Risk Predictor')

model = joblib.load('tnbc_risk_model.joblib')

with st.form('form'):
    age = st.slider('Age', 20, 70)
    race_black = st.selectbox('Are you Black?', ['No', 'Yes'])
    family_history = st.selectbox('Family history of breast cancer?', ['No', 'Yes'])
    early_menstruation = st.selectbox('Started menstruation before age 12?', ['No', 'Yes'])
    late_menopause = st.selectbox('Menopause after age 55?', ['No', 'Yes'])
    obesity = st.selectbox('Obese (BMI > 30)?', ['No', 'Yes'])
    smoking = st.selectbox('Do you smoke?', ['No', 'Yes'])
    breast_pain = st.selectbox('Breast pain?', ['No', 'Yes'])
    palpable_lump = st.selectbox('Lump in breast?', ['No', 'Yes'])
    nipple_discharge = st.selectbox('Nipple discharge?', ['No', 'Yes'])
    skin_dimpling = st.selectbox('Dimpling of breast skin?', ['No', 'Yes'])
    submit = st.form_submit_button('Predict')

if submit:
    data = pd.DataFrame([[
        age,
        1 if race_black == 'Yes' else 0,
        1 if family_history == 'Yes' else 0,
        1 if early_menstruation == 'Yes' else 0,
        1 if late_menopause == 'Yes' else 0,
        1 if obesity == 'Yes' else 0,
        1 if smoking == 'Yes' else 0,
        1 if breast_pain == 'Yes' else 0,
        1 if palpable_lump == 'Yes' else 0,
        1 if nipple_discharge == 'Yes' else 0,
        1 if skin_dimpling == 'Yes' else 0
    ]], columns=[
        "age", "race_black", "family_history", "early_menstruation",
        "late_menopause", "obesity", "smoking", "breast_pain",
        "palpable_lump", "nipple_discharge", "skin_dimpling"
    ])
    pred = model.predict(data)[0]
    st.success("High Risk for TNBC" if pred else "Low Risk for TNBC")
    st.caption("⚠️ Not for medical use.")
