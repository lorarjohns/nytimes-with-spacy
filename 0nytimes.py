#!/usr/bin/env python
# coding: utf-8

# # Reclassifying the news

# In[12]:


from __future__ import unicode_literals

import pickle
import copyreg

import time
from itertools import chain
import tqdm
from tqdm import tqdm_notebook
import spacy
import json
from spacy.tokens import Doc, Token, Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher

from spacy.language import Language

from spacy import displacy 

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

import textacy

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
import string, re, regex

import pandas as pd
import numpy as np

import pymysql.cursors
#from credentials import username, password, host, dbname

import pickle


# In[13]:


from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from gensim.test.utils import common_corpus, common_dictionary
from gensim.sklearn_api import LdaTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score


# In[14]:


import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.models import TfidfModel
import numpy as np

from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint


import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pyLDAvis.sklearn

import pyLDAvis
import pyLDAvis.gensim  # don't skip this


with open('../nytimes_copy/preprocess.pickle', 'rb') as f:
    nlp = pickle.load(f)


nytimes = pd.read_pickle("../nytimes_copy/nytimes_spacy.pickle")

dataframe = pd.read_csv("../nytimes_copy/master_df_R2.csv")

dataframe['Keywords'].apply(lambda x: analyzer.polarity_scores(x))

to_add = dataframe[['Dominant_Topic', 'Topic_Perc_Contrib','Keywords']]


all_news = pd.concat([nytimes, to_add], axis=1)

def to_text(doc, strings=False):
    if not strings:
        return [token.text for token in doc if not token.is_stop]
    else:
        return u" ".join([token.text for token in doc if not token.is_stop])


nytimes['clean_strings'] = nytimes['spacy_docs'].apply(lambda x: to_text(x))

def to_text(doc, strings=False):
    if not strings:
        return [token.text for token in doc if not token.is_stop]
    else:
        return u" ".join([token.text for token in doc if not token.is_stop])

# sklearn version
# docs need to be a list of lists of STRINGS
list_of_str = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))

from gensim.sklearn_api import LdaTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import xgboost
from xgboost import XGBClassifier


from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

union = Pipeline([
    ('vec', Pipeline([ 
        ('tfidf', TfidfVectorizer(use_idf=True)),
            ])),
    ('class', FeatureUnion([
        ('c1', Pipeline([
            ('svd', TruncatedSVD(n_components=300)),
                ])),
        ('c2', Pipeline([
            ('lda', LatentDirichletAllocation()),
        ])),
    ])),
('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
])
parameters = {
 'vec__tfidf__norm': ('l1', 'l2'),
 'vec__tfidf__ngram_range': [(2,3)],
 'class__c2__lda__n_components': [10, 25, 30, 35, 40],
 'class__c2__lda__learning_decay': [.5, .7, .9]
}
gridsearch = GridSearchCV(union, param_grid=parameters, cv=5,
                           n_jobs=-1, verbose=1)
tqdm(gridsearch.fit(list_of_str))

print(gridsearch)

class MyClass:
    def __init__(self, name):
        self.name = name

def pickle_MyClass(obj):
    assert type(obj) is MyClass
    return MyClass, (obj.name,)

copyreg.pickle(MyClass, pickle_MyClass)

if __name__ == '__main__':
    o = MyClass('test')
    with open('out.pkl', 'wb') as f:
        pickle.dump(o, f)

