import streamlit as st
import requests
import pandas as pd
import pickle
import plotly.graph_objs as go

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

fig = go.Figure()
st.write("""
# Sentimental Analysis

""")

st.write('Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("   ")
st.write("   ")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

count_positive = 0
count_negative = 0
count_neutral = 0
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    for i in range(input_df.shape[0]):
        url = 'https://aisentimentsanalyzer.herokuapp.com/classify/?text='+str(input_df.iloc[i])
        r = requests.get(url)
        result = r.json()["text_sentiment"]
        if result=='positive':
            count_positive+=1
        elif result=='negative':
            count_negative+=1
        else:
            count_neutral+=1 

    x = ["Positive", "Negative", "Neutral"]
    y = [count_positive, count_negative, count_neutral]

    if count_positive>count_negative:
        st.write("""# Great Work there! Majority of people liked your product 😃""")
    elif count_negative>count_positive:
        st.write("""# Try improving your product! Majority of people didn't find your product upto the mark 😔""")
    else:
        st.write("""# Good Work there, but there's room for improvement! Majority of people have neutral reactions to your product 😶""")
        
    layout = go.Layout(
        title = 'Multiple Reviews Analysis',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Number of reviews'),)
    
    fig.update_layout(dict1 = layout, overwrite = True)
    fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("# ⬅ Enter user input from the sidebar to see the nature of the review.")

pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(uploaded_file)
