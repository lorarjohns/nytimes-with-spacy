from __future__ import unicode_literals
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
from credentials import username, password, host, dbname

import pickle

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from gensim.test.utils import common_corpus, common_dictionary
from gensim.sklearn_api import LdaTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score

import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.models import TfidfModel
import numpy as np

from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint

import textacy
from textacy import tm
from textacy.vsm import Vectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation


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
%matplotlib notebook

from gensim.test.utils import common_corpus, common_dictionary
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

from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
