#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:27:37 2020

@author: kiran
"""

import pandas as pd
import re
import nltk

nltk.download('stopwords')

from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


messages =pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t', names=['review','message'])
#dataset_2 = pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t',header = None, names = ['res', 'message'])

corpus = []

ps = PorterStemmer()
#lemmatizer = WordNetLemmatizer()

for i in range(len(messages)):
    
    res = re.sub('[^a-zA-Z]',' ', messages['message'][i])
    res = res.lower()
    res = res.split()
    res = [ps.stem(word) for word in res if word not in set(stopwords.words('english'))]
    res = " ".join(res)
    corpus.append(res)    
    
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

cv = CountVectorizer(max_features= 5000)
X = cv.fit_transform(corpus).toarray()

y = messages.iloc[:, :-1]
y = pd.get_dummies(y['review'])
y = y.iloc[:, 1].values
y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.20, random_state = 0)
 
  
# from sklearn.naive_bayes import GaussianNB

# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)



from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

  
