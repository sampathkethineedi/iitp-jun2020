import requests
import streamlit as st
import json

st.title('Summarizer')
num_beams = st.sidebar.slider("Number of beams", min_value=2, max_value=32)

article = st.text_input("Add article to generate summary of")
input_dict = {'article': article, 'num_beams': num_beams}

if st.button("Generate Summary"):
    response = requests.post('http://127.0.0.1:5000/summapi/v1.0.0/summ', json=input_dict)
    if(response.status_code == 201):
        st.write("Summarization Complete")
        data = response.json()
        st.write(data['summary'][0])
    else:
        st.write(f"Error {response.status_code}")

if st.button("Generate Extreme Summary (One Line Summary)"):
    response = requests.post('http://127.0.0.1:5000/summapi/v1.0.0/xsumm', json=input_dict)
    if(response.status_code == 201):
        st.write("Extreme Summarization Complete")
        data = response.json()
        st.write(data['summary'][0])
    else:
        st.write(f"Error {response.status_code}")
