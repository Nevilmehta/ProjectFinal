import streamlit as st
from home import main as home_main
from contact import main as contact_main
from defaultdata import main as defaultdata_main
from tweetdata import main as twitterdata_main
from youtubedata import main as youtubedata_main
from emotion_main import main as emotiondata_main

# st.write("""
# # Sentiment Analysis

# """)

def main():

    # st.write('Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.')
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.header('Sentiment Types')
    menu = ["Home","Default","Twitter","YouTube","Emotion","Contact"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice== "Home":
        home_main()
    elif choice== "Default":
        defaultdata_main()
        # pass
    elif choice== "Twitter":
        twitterdata_main()        
    elif choice== "YouTube":
        youtubedata_main()
    elif choice== "Emotion":
        emotiondata_main()
    elif choice== "Contact":
        contact_main()

    st.sidebar.subheader("""RejoiceHub Solutions """)
    st.sidebar.image('logo.jpg', width = 300)

main()
# st.sidebar.image('logo.jpg', width = 300)
