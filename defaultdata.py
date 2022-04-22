import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
from nltk.stem import WordNetLemmatizer
from server.utils.default import CSV_File_pred
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0:
			pos_list.append(i)
			pos_list.append(res)

		elif res < 0:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

def get_df(file):
  # get extension and read file
    extension = file.name.split('.')[1]
    if extension.upper() == 'CSV':
        df = pd.read_csv(file)
    elif extension.upper() == 'XLSX':
        df = pd.read_excel(file, engine='openpyxl')
    elif extension.upper() == 'PICKLE':
        df = pd.read_pickle(file)
    return df

def main():
	st.title("Sentiment Analysis")
	st.subheader("For simple Text and default dataset")

	menu = ["Home","Default data"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("Results")
				sentiment = TextBlob(raw_text).sentiment
				st.write(sentiment)

				if sentiment.polarity > 0:
					st.markdown("Sentiment:: Positive :smiley: ")
				elif sentiment.polarity < 0:
					st.markdown("Sentiment:: Negative :angry: ")
				else:
					st.markdown("Sentiment:: Neutral 😐 ")

				result_df = convert_to_df(sentiment)
				st.dataframe(result_df)

				c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
				st.altair_chart(c,use_container_width=True)

			with col2:
				st.info("Token Sentiment")

				token_sentiments = analyze_token_sentiment(raw_text)
				st.write(token_sentiments)

	elif choice=="Default data":
		st.subheader("Default data")

		
		st.write('A general purpose data exploration app')
		file = st.file_uploader("Upload file", type=['csv' 
												,'xlsx'
												,'pickle'])
		
		if file:
			df= pd.read_excel(file)
			st.write("CSV")
			st.dataframe(df['text'])
			st.write(CSV_File_pred(df))

		
		# if not file:
		# 	st.write("Upload a .csv or .xlsx file to get started")
			
		# 	return 

		# df = get_df(file)
		# dff= normalizer(df)
		# return dff

# main()