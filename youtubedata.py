import os
import re
import googleapiclient
import googleapiclient.discovery
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from textblob import TextBlob
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


def google_api(id):
    
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyAoNLmo6pMjBkW2CzXOeOtKzp3N_RTjOOY"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=100,
        order="relevance",
        videoId= id
    )
    response = request.execute()

    return response

def CSV_File_pred(df):
      
      def normalizer(tweett):

            only_letters = re.sub("[^a-zA-Z]", " ", tweett)
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
      x=pd.Series(df['Comments'])
      x_vectorized = vectorizer.fit_transform(x.apply(normalizer))

      train_x,val_x,train_y,val_y = train_test_split(x_vectorized,df['Analysis'])

      regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
      model = regressor.fit(train_x, train_y)

      params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
      gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
      gs_clf = gs_clf.fit(train_x, train_y)
      model = gs_clf.best_estimator_
      y_pred = model.predict(val_x)

      score = accuracy_score(val_y, y_pred)
      _f1 = f1_score(val_y, y_pred, average='micro')
      __precision = precision_score(val_y, y_pred, average='micro')
      _recall = recall_score(val_y, y_pred, average='micro')

      test_feature = vectorizer.transform(['Meat Week Day 3: Tummy hurts every night'])
      x=model.predict(test_feature)
      print(x)
      return f'F1-Score: {_f1} | Accuracy: {score} | Sentiment: {x}'

def create_df_author_comments(response):

    authorname = []
    comments = []

    for i in range(len(response["items"])):
        authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
        comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
    
    df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
    return df_1 

def cleaning_comments(comment):

    comment = re.sub("[0-9]+","",comment)
    comment = re.sub("[\:|\@|\)|\*|\.|\$|\!|\?|\,|\%|\"]+"," ",comment)
    comment = re.sub("[\(|\-|\”|\“|\#|\!|\/|\«|\»|\&]+","",comment)
    comment = re.sub("\n"," ",comment)
    comment = re.sub('[\'|🇵🇰|\;|\!]+','',comment)
    return comment

def remove_comments(df_1):
  
    zero_length_comments = df_1[df_1["Comments"].map(len) == 0]
    zero_length_comments_index = [ind for ind in zero_length_comments.index] 
    df_1.drop(zero_length_comments_index, inplace = True)
    return df_1

def find_subjectivity_on_single_comment(text):
    return TextBlob(text).sentiment.subjectivity
   
def apply_subjectivity_on_all_comments(df_1):
    df_1['Subjectivity'] = df_1['Comments'].apply(find_subjectivity_on_single_comment)
    return df_1 

def find_polarity_of_single_comment(text):
    return TextBlob(text).sentiment.polarity

def find_polarity_of_every_comment(df_1):  
    df_1['Polarity'] = df_1['Comments'].apply(find_polarity_of_single_comment)
    return df_1 

def analysis_based_on_polarity(df_1):
    df_1['Analysis'] = df_1['Polarity'].apply(analysis)
    return df_1


# Lambda functions
lower = lambda comment: comment.lower()
analysis = lambda polarity: 'Positive' if polarity > 0 else 'Neutral' if polarity == 0 else 'Negative' 

# Program flow
# response = google_api(id)

def analyse_comments(df):

    df_1 = df.copy()
    df_1["Comments"]= df_1["Comments"].apply(cleaning_comments)
    df_1['Comments'] = df_1['Comments'].apply(lower)
    df_1 = remove_comments(df_1)
    df_1 = apply_subjectivity_on_all_comments(df_1)
    df_1 = find_polarity_of_every_comment(df_1)
    df_1 = analysis_based_on_polarity(df_1)
    return df_1

def main():

    st.title('Live Youtube video comments Sentiment Analysis')
    st.markdown('This app uses GoogleclientApi to get youtube comments from YouTube video')

    with st.form(key='nlpForm'):
        video_id = st.text_input("Enter the Id of youtube Video")
        submit_button = st.form_submit_button(label='Analyze')

    if submit_button:
        print("\n\n\n\n")
        print(video_id)
        print("\n\n\n\n")
        response =google_api(video_id)
        df_1 = create_df_author_comments(response)
        df = analyse_comments(df= df_1)
        st.info("Results")
        st.write(df)
        st.write(CSV_File_pred(df))
    
# main()