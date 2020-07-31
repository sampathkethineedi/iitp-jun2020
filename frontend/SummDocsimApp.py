import requests
import streamlit as st
import json

st.title('Summarizer')
num_beams = st.sidebar.slider("Number of beams", min_value=2, max_value=32)

article = st.text_input("Add article to generate summary of")
input_dict = {'article': article, 'num_beams': num_beams}

url = 'http://127.0.0.1:5000'

if st.button("Generate Summary"):
    response = requests.post(f'{url}/summapi/v2.0.0/summ', json=input_dict)
    if(response.status_code == 201):
        st.write("Summarization Complete")
        data = response.json()
        st.write(data['summary'][0])
    else:
        st.write(f"Error {response.status_code}")

if st.button("Generate Extreme Summary (One Line Summary, may be useful only for short articles)"):
    response = requests.post(f'{url}/summapi/v2.0.0/xsumm', json=input_dict)
    if(response.status_code == 201):
        st.write("Extreme Summarization Complete")
        data = response.json()
        st.write(data['summary'][0])
    else:
        st.write(f"Error {response.status_code}")

cos_threshold = st.sidebar.slider("Cosine Similarity Threshold", min_value=0.0, max_value=0.99)

if st.button("Get similar articles"):
    payload = {'cos_tsld': cos_threshold}
    response = requests.get(f'{url}/summapi/v2.0.0/docsim', params=payload)
    if(response.status_code == 200):
        st.write("Similar articles and scores:")
        data = response.json()
        for idx in range(len(data["data"])):
            sim_article = data["data"][idx]
            st.write(f"Article: {sim_article}")
            score = (data["scores"][idx])/100
            st.write(f"Score: {score}")
    else:
        st.write(f"Error {response.status_code}")        
