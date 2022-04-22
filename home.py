import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding= load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_n0b8rrnw.json")

def main():

  with st.container():
      st.subheader("Hello! Welcome to sentiment")
      st.title("What is Sentiment Analysis?")
      st.write("Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral.")
      st.write("A sentiment analysis system for text analysis combines natural language processing (NLP) and machine")         
      st.write("learning techniques to assign weighted sentiment scores to the entities, topics, themes and categories") 
      st.write("within a sentence or phrase.")
      st.write("[Learn More >](https://monkeylearn.com/sentiment-analysis/)")

  with st.container():
      st.write("---")
      left_column, right_column= st.columns(2)
      with right_column:
          st.header("About")
          st.write("##")
          st.write(
              """
              Various kind of Sentimental analysis:
              - Twitter's Tweets analysis and report
              - Youtube's Comments analysis and report
              - Emotion analysis on expressive blogs
              - Social media review analysis with twitter and youtube response
              - Default data analysis with uploading data 

              For grow of your business, keep doing prediction and save time.
              """
          )
      
      with left_column:
          st_lottie(lottie_coding, height=300, key="coding")

# main()
