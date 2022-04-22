import os
import re
import googleapiclient
import googleapiclient.discovery
import pandas as pd
import pickle
from textblob import TextBlob

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
response = google_api("621oD2zBSbI")

def create_df_author_comments():

    authorname = []
    comments = []

    for i in range(len(response["items"])):
        authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
        comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
    
    df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
    return df_1 

df_1 = create_df_author_comments()

#cleaning
def cleaning_comments(comment):

    comment = re.sub("[0-9]+","",comment)
    comment = re.sub("[\:|\@|\)|\*|\.|\$|\!|\?|\,|\%|\"]+"," ",comment)
    comment = re.sub("[\(|\-|\”|\“|\#|\!|\/|\«|\»|\&]+","",comment)
    comment = re.sub("\n"," ",comment)
    comment = re.sub('[\'|🇵🇰|\;|\!]+','',comment)
    return comment

#lower
df_1["Comments"]= df_1["Comments"].apply(cleaning_comments)
lower = lambda comment: comment.lower()
df_1['Comments'] = df_1['Comments'].apply(lower)

#null spaces comments
def remove_comments(df_1):
  
    zero_length_comments = df_1[df_1["Comments"].map(len) == 0]
    zero_length_comments_index = [ind for ind in zero_length_comments.index] 
    df_1.drop(zero_length_comments_index, inplace = True)
    return df_1

df_1 = remove_comments(df_1)

def find_subjectivity_on_single_comment(text):
    return TextBlob(text).sentiment.subjectivity
   
def apply_subjectivity_on_all_comments(df_1):
    df_1['Subjectivity'] = df_1['Comments'].apply(find_subjectivity_on_single_comment)
    return df_1 

df_1 = apply_subjectivity_on_all_comments(df_1)

#find polarity
def find_polarity_of_single_comment(text):
    return TextBlob(text).sentiment.polarity

def find_polarity_of_every_comment(df_1):  
    df_1['Polarity'] = df_1['Comments'].apply(find_polarity_of_single_comment)
    return df_1 

df_1 = find_polarity_of_every_comment(df_1)

analysis = lambda polarity: 'Positive' if polarity > 0 else 'Neutral' if polarity == 0 else 'Negative' 

def analysis_based_on_polarity(df_1):
    df_1['Analysis'] = df_1['Polarity'].apply(analysis)
    return df_1
  
df_1 = analysis_based_on_polarity(df_1)
print(df_1)
