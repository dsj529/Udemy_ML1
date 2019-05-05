'''
Created on Apr 23, 2019

@author: dsj529

revised version of k-means text clustering example in online course.
'''
from collections import defaultdict
from pprint import pprint

from nltk import word_tokenize, pos_tag
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


TAG_MAP = defaultdict(lambda: wn.NOUN)
TAG_MAP['J'] = wn.ADJ
TAG_MAP['V'] = wn.VERB
TAG_MAP['R'] = wn.ADV
TAG_MAP['S'] = wn.ADJ_SAT

def tokenizer(text):
        tokens = word_tokenize(text)
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(token, TAG_MAP[tag[0]]) 
                  for token, tag in pos_tag(tokens) 
                      if token not in stopwords.words('english')]
        return tokens


def cluster_sentences(sentences, nb_of_clusters=2): 

        # create tf ifd again: stopwords-> we filter out common words (I,my, the, and...)
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'),lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return dict(clusters)


if __name__ == "__main__":
        sentences = ["Quantuum physics is quite important in science nowadays.",
                "Investing in stocks and trading with them are not that easy",
                "FOREX is the stock market for trading currencies",
                "Software engineering is hotter and hotter topic in the silicon valley",
                "Warren Buffet is famous for making good investments. He knows stock markets"]
        nclusters= 2
        clusters = cluster_sentences(sentences, nclusters)
        for cluster in range(nclusters):
                print("CLUSTER ",cluster,":")
                for i,sentence in enumerate(clusters[cluster]):
                    print("\tSENTENCE ",i,": ",sentences[sentence])