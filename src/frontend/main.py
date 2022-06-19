from backend.engine import *
import config
from backend import inference

import streamlit as st
import datetime
hour = datetime.datetime.now().hour

greeting = "Good Morning" if 5<=hour<12 else "Good Afternoon" if 12<=hour<18 else "Good Evening"

# Set up webframe
print("Printing titles...")
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", initial_sidebar_state="auto")
st.title(config.PAGE_TITLE)
st.write(f"### {greeting} ! ")


if "models_engine" not in st.session_state:
    st.session_state.models_engine = False
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Load models
if not st.session_state.models_engine:
    models_engine = {}
    print("Models loading in progress...")
    models_checkpoints = config.MODEL_CHKPT_MAP
    for model, checkpoint in models_checkpoints.items():
        if model == "facebook/bart-base":
            models_engine["BART-base"] = BartBaseEngine(checkpoint)
        if model == "facebook/bart-large":
            models_engine["BART-large"] = BartLargeEngine(checkpoint)
        if model == "t5-base":
            models_engine["T5-base"] = T5BaseEngine(checkpoint)
        if model == "t5-small":
            models_engine["T5-small"] = T5SmallEngine(checkpoint)
        if model == "google/t5-v1_1-small":
            models_engine["T5-v11"] = T5V11Engine(checkpoint)

    st.session_state.models_engine = models_engine
    print("Model loading completed.")
    print(st.session_state.models_engine)

# Model selection
model_list = ["BART-base", "BART-large", "T5-base", "T5-small", "T5-v11"]
model_name = st.sidebar.selectbox("Select Model", model_list)

# Document input
document = st.text_area("Input Text to be summarized", height=250)

# Slide bars
max_length = st.sidebar.slider("Summary MAX length", 50, 250)
min_length = st.sidebar.slider("Summar MIN length", 10, 100)
beam_size = st.sidebar.slider("Beam size configuration", 2, 12)

# Summary output
st.write("_"*40)
st.write("### Summary result:")
st.write(f"Model selected: **{model_name}**")

if st.button("GENERATE"):
    print("Generating summary...")
    print(st.session_state.models_engine)
    
    st.session_state.summary = st.session_state.models_engine[model_name].summarize(document, max_length, min_length, beam_size)
    print("Summary generated.")
st.write(st.session_state.summary)