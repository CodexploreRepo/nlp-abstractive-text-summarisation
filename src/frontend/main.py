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
model_list = ["BART", "T5"]
model_name = st.sidebar.selectbox("Select Model", model_list)
st.write(f"Model selected: **{model_name}**")

# Document input
document = st.text_area("Input Text to be summarized")

# Slide bars
words_in_summary = st.sidebar.slider("#words in summary", 10, 100)
beam_size = int(st.sidebar.slider("beam size", 2, 12))

# Summary output
st.write("_"*40)
st.write("### Summary result:")
if "summary" not in st.session_state:
    st.session_state.summary = ""
if st.button("GENERATE"):
    st.session_state.summary = inference.summarize(document, words_in_summary, beam_size)
    st.text(st.session_state.summary)
else:
    st.text(st.session_state.summary)

# # get top artist
# st.write(f"Most Popular Artist in {config.COUNTRY}")
# top_artist = song_scraper.get_top_artist_by_geo("Singapore")
# if 'Error' in top_artist:
#     st.write("Can't fetch any artists")
# else:
#     for tr in top_artist:
#         url = tr['url']
#         st.markdown(f"[- {tr['name']}](%s)" % url)


# # get top tracks
# st.write(f"Most Popular Song in {config.COUNTRY}")
# top_song = song_scraper.get_top_track_by_geo("Singapore")
# if 'Error' in top_song:
#     st.write("Can't fetch any tracks")
# else:
#     for tr in top_song:
#         url = tr['url']
#         st.markdown(f"[- {tr['name']}](%s)" % url)
