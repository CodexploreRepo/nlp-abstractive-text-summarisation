import config
from backend import inference

import streamlit as st
import datetime
hour = datetime.datetime.now().hour

greeting = "Good Morning" if 5<=hour<12 else "Good Afternoon" if 12<=hour<18 else "Good Evening"

# Set up webframe
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", initial_sidebar_state="auto")
st.title(config.PAGE_TITLE)
st.write(f"### {greeting} ! ")

# Model selection
model_list = ["BART-base", "BART-large", "T5-base", "T5-small"]
model_name = st.sidebar.selectbox("Select Model", model_list)

# Document input
document = st.text_area("Input Text to be summarized")

# Slide bars
words_in_summary = st.sidebar.slider("#words in summary", 10, 100)
beam_size = int(st.sidebar.slider("beam size", 2, 12))

# Summary output
st.write("_"*40)
st.write("### Summary result:")

st.write(f"Model selected: **{model_name}**")

if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.button("GENERATE"):
    st.session_state.summary = inference.engine(document, model_name, words_in_summary, beam_size)

st.text(st.session_state.summary)