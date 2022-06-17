#from backend.inference import HELLO
import config

import streamlit as st
import datetime
hour = datetime.datetime.now().hour

greeting = "Good Morning" if 5<=hour<12 else "Good Afternoon" if 12<=hour<18 else "Good Evening"

# Set up webframe
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", initial_sidebar_state="auto")

st.title(config.TITLE)
st.write(f"""
    ### {greeting} ! 
    #### Personalized Music Recommendation
""")

# Model selection
model_list = ["BART", "T5"]
model_name = st.sidebar.selectbox("Select Model", model_list)


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
