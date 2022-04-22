from tweepy import Stream

from tweepy import OAuthHandler 
from tweepy.streaming import StreamListener

consumer_key="iVAQ0Z9FkIXQF9AW039FiPgXv"

consumer_secret="a113DEJJznVf6yX5xBaUXjMwOECGe60FCbSQCirrbwA0q0QBKT"

access_token="1369575677377327106-suWKbtyRHTfYnbe1ZJY1wthSAInYqi"

access_secret="nFxC5wpxaTKIKo7ZpDR8VIsXEtI7xVNzULj5ZGTxwPPgF"

class TweetListener(StreamListener): 
  def on_data(self, data): 
    print (data) 
    return True

  def on_error(self, status):
    print (status)

auth =OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_secret)

stream =Stream(auth, TweetListener()) 
stream.filter(track=['#putin'])