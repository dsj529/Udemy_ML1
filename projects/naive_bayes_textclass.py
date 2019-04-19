'''
Created on Apr 14, 2019

@author: dsj529
'''
from collections import defaultdict

from matplotlib import pyplot as plt
from nltk import word_tokenize, pos_tag
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import numpy as np


TAG_MAP = defaultdict(lambda: wn.NOUN)
TAG_MAP['J'] = wn.ADJ
TAG_MAP['V'] = wn.VERB
TAG_MAP['R'] = wn.ADV
TAG_MAP['S'] = wn.ADJ_SAT

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, text):
        tokenized = word_tokenize(text)
        return [self.wnl.lemmatize(token, TAG_MAP[tag[0]])
                for token, tag in pos_tag(tokenized)]
        
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

print('defining dataset')
trainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
countVectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                  lowercase=True,
                                  strip_accents='unicode')
print('transforming data to tfidf')
xTrainCounts = countVectorizer.fit_transform(trainingData.data)
print('done')
# print(countVectorizer.vocabulary_.get(u'software'))

tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

model = MultinomialNB().fit(xTrainTfidf, trainingData.target)
preds = model.predict(xTrainTfidf)
print(confusion_matrix(trainingData.target, preds))
print(accuracy_score(trainingData.target, preds))
print(classification_report(trainingData.target, preds))

new = ['This has nothing to do with church or religion', 
       'Software engineering is getting hotter and hotter nowadays']

xNewCounts = countVectorizer.transform(new)
xNewTfidf = tfidTransformer.transform(xNewCounts)

predicted = model.predict(xNewTfidf)

for doc, category in zip(new,predicted):
    print('%r --------> %s' % (doc, trainingData.target_names[category]))
