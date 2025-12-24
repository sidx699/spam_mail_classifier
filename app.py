import streamlit as st
from joblib import load

st.set_page_config(
    page_title="Spam Email Classifier",
    layout='centered'
)

def load_model():
    return load("model/spam_model.joblib")

st.title("Spam Email Classifier")

text=st.text_area(placeholder="Enter a message here to check whether it is spam or not.",label="")

if st.button("Check", type="primary"):
    if text.strip() == "":
        st.warning("Please enter a message.")
    else:
        model=load_model()
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0][1]
        if prediction == 1:
            st.error(f"Spam detected (Probability: {probability:.2%})")
        else:
            st.success(f"Not Spam (Probability: {1 - probability:.2%})")
