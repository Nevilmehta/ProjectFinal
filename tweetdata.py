import tweepy as tw
import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,precision_score,recall_score
wordnet_lemmatizer = WordNetLemmatizer()


consumer_key="iVAQ0Z9FkIXQF9AW039FiPgXv"
consumer_secret="a113DEJJznVf6yX5xBaUXjMwOECGe60FCbSQCirrbwA0q0QBKT"
access_token="1369575677377327106-suWKbtyRHTfYnbe1ZJY1wthSAInYqi"
access_secret="nFxC5wpxaTKIKo7ZpDR8VIsXEtI7xVNzULj5ZGTxwPPgF"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = pipeline('sentiment-analysis')


def CSV_File_pred(df):
      
      def normalizer(tweet):

            only_letters = re.sub("[^a-zA-Z]", " ", tweet)
            only_letters = only_letters.lower()
            only_letters = only_letters.split()
            filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
            lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
            lemmas = ' '.join(lemmas)
            return lemmas

      # df = shuffle(df)
      # y = df['airline_sentiment']
      # x = df.text.apply(normalizer)
      # print(x.head())

      vectorizer = CountVectorizer()
      print(df.columns)
      x=pd.Series(df['text'])
      x_vectorized = vectorizer.fit_transform(x.apply(normalizer))

      train_x,val_x,train_y,val_y = train_test_split(x_vectorized,df['sentiment'])

      regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
      model = regressor.fit(train_x, train_y)

      params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
      gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
      gs_clf = gs_clf.fit(train_x, train_y)
      model = gs_clf.best_estimator_
      y_pred = model.predict(val_x)

      _f1 = f1_score(val_y, y_pred, average='micro')
      score = accuracy_score(val_y, y_pred)
      __precision = precision_score(val_y, y_pred, average='micro')
      _recall = recall_score(val_y, y_pred, average='micro')

      test_feature = vectorizer.transform(['Meat Week Day 3: Tummy hurts every night'])
      x=model.predict(test_feature)
      print(x)
      return f'Overall Sentiment of tweets: {x}'


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

def main():
    st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
    st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

    with st.form(key="Enter name"):
        search_words = st.text_input("Enter the name for which you want to know the sentiment")
        number_of_tweets = st.number_input("Enter the number of latest tweets for which you want to know the sentiment", 0,50,10)
        submit_button = st.form_submit_button(label="Submit")
    
    if submit_button:
        tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
        tweet_list = [i.text for i in tweets]
        p = [i for i in classifier(tweet_list)]
        q=[p[i]["label"] for i in range(len(p))]
        text= "text"
        df = pd.DataFrame(list(zip(tweet_list, q)),columns =[text, "sentiment"])
        st.write(df)
        st.write(CSV_File_pred(df))
    
# main()