import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,precision_score,recall_score
wordnet_lemmatizer = WordNetLemmatizer()

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

      train_x,val_x,train_y,val_y = train_test_split(x_vectorized,df['airline_sentiment'])

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
      return f'F1-Score: {_f1} | Accuracy: {score} | Sentiment: {x}'

# print(CSV_File_pred('B:/intern/Sentiment analyse/Tweets.csv'))
# test_feature = vectorizer.transform(['Movie is good'])
# model.predict(test_feature)

# test_feature = vectorizer.transform(['I\'m okay'])
# model.predict(test_feature)
'''
 pickle.dump(model, open('model.pkl', 'wb'))
'''