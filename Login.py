# Login.py
import streamlit as st
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title=config["app"]["title"], layout="wide", page_icon="📊")

if "auth" not in st.session_state:
    st.session_state.auth = False

if st.session_state.auth:
    st.switch_page("pages/Dashboard.py")

st.title("Login")
password = st.text_input("Password", type="password")
if st.button("Login"):
    if password == config["app"]["password"]:
        st.session_state.auth = True
        st.success("Logged in successfully!")
        st.switch_page("pages/Dashboard.py")
    else:
        st.error("Incorrect password")
