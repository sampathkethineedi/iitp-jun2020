import streamlit as st
import requests as req

CHOICES = {0: "Select an option", 1: "Recognise", 2: "Register", 3: "Delete", 4: "Authenticate"}


def action(option):
    return CHOICES[option]

def recognise():
    URL = "http://127.0.0.1:5000/recognise"
    r = req.get(URL).json()
    data = r['rec spk']
    if(data[0]=='MATCH!'):
        st.write("MATCH FOUND!!!")
        st.write(data[1])
    else:
        st.write(data[0])
    return
    
def register():
    URL = "http://127.0.0.1:5000/register"
    spk_name = st.text_input("Name of the speaker to be registerd: ")
    if(st.button("Enter")):
        if(len(spk_name)==0):
            st.error("Enter a valid name")
        else:
            body = {"speaker": spk_name}
            r = req.post(URL,json=body)
            st.markdown("<h4 style='color: green;'>SUCCESSFULLY REGISTERED</h4>" + spk_name, unsafe_allow_html=True)
            
    return
       
def delete():
    URL = "http://127.0.0.1:5000/delete"
    spk_name = st.text_input("Name of the speaker to be deleted: ")
    if(st.button("Enter")):
        if(len(spk_name)==0):
            st.error("Enter a valid name")
        else:
            body = {"speaker": spk_name}
            r = req.post(URL,json=body).json()
            data = r['del']
            st.write(data)            
    return

def authenticate():
    URL = "http://127.0.0.1:5000/authenticate"
    spk_name = st.text_input("Enter your name: ")
    if(st.button("Enter")):
        if(len(spk_name)==0):
            st.error("Enter a valid name")
        else:
            body = {"speaker": spk_name}
            r = req.post(URL,json=body).json()
            data = r['auth_spk']
            st.write(data)            
    return
    

def main():
    st.markdown("<h1 style='text-align: center; color: red;'>SPEAKER RECOGNITION</h1>", unsafe_allow_html=True)
    option = st.selectbox("What do you want to do?", options=list(CHOICES.keys()),format_func=action)
    if(option==0):
        st.write("Waiting for your choice...")
    else:
        st.write(f"You chose to {action(option)}")
        if(option==1):
            recognise()
        elif(option==2):
            register()
        elif(option==3):
            delete()
        else:
            authenticate()
    

    

if __name__=="__main__":
    main()

