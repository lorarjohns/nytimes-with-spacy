    #!/usr/bin/env python
    # coding: utf-8

    # # Reclassifying the news

    # In[12]:


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


    # In[5]:


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
    get_ipython().run_line_magic('matplotlib', 'notebook')


    # In[3]:


    nytimes = pd.read_csv('csv/nytimes.csv')
    nytimes.columns


    # In[389]:


    nytimes.info()


    # In[390]:


    nytimes.head()


    # In[391]:


    nytimes.reset_index(inplace=True, drop=True, col_level=0)


    # In[392]:


    nytimes.drop(['Unnamed: 0', 'id', 'stems'], axis=1, inplace=True)


    # In[393]:


    nytimes.head()


    # In[394]:


    print(nytimes['url'][0])


    # In[395]:


    topic_re = r"(?<=\d{4}\/\d{2}\/\d{2}\/)(\w+\-?\w?\/?\w+\/)"


    # In[396]:


    topics = nytimes['url'].str.extract(topic_re, expand=True)


    # In[397]:


    # only one NaN
    na = topics[topics[0].isnull()]
    na


    # In[398]:


    topics.dropna(inplace=True)


    # In[399]:


    nytimes.drop(index=3886, inplace=True)


    # In[400]:


    topics[0] = topics[0].map(lambda x: x.replace('/', ' '))
    pattern = re.compile("(\w+) (\w+)?")
    add_topics = topics[0].str.extract("(\w+) (\w+)?", expand=True)


    # In[401]:


    nytimes = pd.concat([nytimes, add_topics], axis=1)


    # In[402]:


    # see date range
    date = pd.to_datetime(nytimes['date'])
    date = pd.DataFrame(date)


    # In[403]:


    #NB: We may want to use this later
    date = pd.DataFrame(date)
    print(date.groupby(date['date'].dt.year).count())


    # In[404]:


    cols = {0 : "topic0", 1 : "topic1"}
    nytimes.rename(columns=cols, inplace=True)


    # In[405]:


    nytimes.head()


    # In[406]:


    len(nytimes['content'])


    # # How many topics are there?
    # 
    # ### And how many should we try to predict?

    # In[407]:


    topics = [x for x in [list(nytimes['topic0'].unique()), list(nytimes['topic1'].unique())]]


    # In[408]:


    set(topics)


    # In[409]:


    # How many topics?
    nytimes['topic0'].unique()


    # In[410]:


    nytimes['topic1'].unique()


    # In[411]:


    topicsdf = pd.concat([nytimes['topic0'],nytimes['topic1']], axis=0)
    topicsdf.fillna("")
    topicsdf = pd.DataFrame(topicsdf)
    topicsdf.columns = ["all_topics"]


    # In[412]:


    topicsdf.dropna(inplace=True)


    # In[417]:


    sns.set(style="darkgrid")
    sns.set(rc={'figure.figsize':(15,9)})
    sns.countplot(x=topicsdf["all_topics"], data=topicsdf)
    plt.xticks(rotation=75, horizontalalignment='right')
    plt.tight_layout()
    sns.despine()


    # In[ ]:


    fig, ax = plt.subplots(1,2)
    sns.set(rc={'figure.figsize':(30,25)})
    sns.countplot(nytimes['topic0'], ax=ax[0])
    ax_twin = ax.twinx()

    sns.countplot(nytimes['topic1'], ax=ax[1])
    plt.xticks(rotation=75, horizontalalignment='right')
    fig.show()


    # ## EDA - Try it out on one thing

    # In[ ]:


    # First, try code out with ONE row in perfect format!


    # In[350]:


    # TRYING IT OUT
    nlp = spacy.load('en_core_web_lg') # loading the language model 
    doc = nlp(nytimes['content'][0])


    # In[348]:


    nlp.pipe_names


    # In[349]:


    doc


    # In[346]:


    sentences = list(doc.sents) 


    # In[ ]:


    displacy.serve(sentences, style="dep")


    # In[ ]:


    displacy.render(doc, style="ent")


    # In[ ]:


    for token in doc: 
    print(f"TOKEN: {token}\t\t\t POS:{token.pos_}")


    # In[ ]:


    span = doc[0:15]
    span[9].is_stop # "an"
    span[1].pos_


    # In[ ]:


    def count_stops(doc):
    stops = []
    count = 0
    for token in doc:
        if token.is_stop == True:
            stops.append(token.text)
            count += 1
    print(f"NUMBER OF STOPS: {count}")
    print(f"STOP WORDS: {set(stops)}")
    count_stops(doc)


    # # Remove stopwords

    # In[39]:


    nlp = spacy.load('en_core_web_lg')

    stopwords = spacy.lang.en.STOP_WORDS
    stop_list = ["Mr.","Mrs.","Ms.","say","WASHINGTON","'s"]
    nlp.Defaults.stop_words.update(stop_list)


    # In[40]:


    def update_stop_words(new_word_list):
    for lex in new_word_list:
         #if lex not in spacy.lang.en.STOP_WORDS:
        nlp.Defaults.stop_words.update(lex)
    #     for word in STOP_WORDS:
    #         lexeme = nlp.vocab[word]
    #         lexeme.is_stop = True

    for word in nlp.Defaults.stop_words:
        for w in (word, word[0].upper() + word[1:], word.upper()):
            lexeme = nlp.vocab[w]
            lexeme.is_stop = True


    # In[41]:


    update_stop_words(stop_list)


    # In[42]:


    for word in stop_list:
    print(nlp.vocab[word].is_stop)


    # In[43]:


    def cleaner(doc):
    doc = [token.text for token in doc if (not token.is_stop and not token.is_punct)]
    doc = u' '.join(doc)
    return nlp.make_doc(doc)

    nlp.add_pipe(cleaner, name="cleaner", first="true")


    # In[44]:


    doc = nlp(nytimes['content'][0])
    with open('preprocess.pickle', 'wb') as f:
    pickle.dump(nlp, f)


    # In[45]:


    for token in doc:
    print(f"TOKEN: {token}\t\tPOS: {token.pos_}")


    # In[46]:


    doc = nlp(nytimes['content'][0])
    doc


    # # Bag of Words

    # In[311]:


    # load the nlp preprocessor
    with open('preprocess.pickle', 'rb') as f:
    nlp = pickle.load(f)


    # In[50]:


    def lemmatizer(doc):
    doc = [token.lemma_ for token in doc if not token.lemma_ == '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)

    nlp.add_pipe(lemmatizer, 'lemmatizer', after='tagger')


    # ## Convert spaCy to text and add to DataFrame for sklearn and gensim

    # In[51]:


    # VERY BIG AND DANGEROUS CELL TO RUN
    # DO NOT DO IT
    # SERIOUSLY

    with open('preprocess.pickle', 'rb') as f:
    nlp = pickle.load(f)

    nlp.add_pipe(lemmatizer,name='lemmatizer',after='ner')
    nlp.pipe_names

    # pickle this nlp pipeline
    with open('all_pipes.pickle', 'wb') as f:
    pickle.dump(nlp, f)

    docs = nytimes['content'].tolist()
    spacy_docs = []

    for doc in tqdm_notebook(nlp.pipe(docs)):    
    doc = [token.text for token in doc]
    doc = u' '.join(doc)
    spacy_docs.append(nlp.make_doc(doc))

    spacy_df = pd.DataFrame([]).append([spacy_docs]).T
    spacy_df.columns = ['spacy_docs']

    # DO NOT SKIP UNLESS YOU WANT NANS

    spacy_df.reset_index(drop=True, inplace=True)
    nytimes.reset_index(drop=True, inplace=True)

    pd.concat([nytimes, spacy_df], axis=1)
    # import os
    # nytimes.to_pickle('nytimes_spacy.pickle')
    # nytimes.to_csv('nytimes_spacy.csv')


    # In[7]:


    # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    # analyzer = SentimentIntensityAnalyzer()
    # 
    # nytimes.drop(['sentiment'], axis=1, inplace=True)


    # In[337]:


    # nytimes['sentiment'] = dataframe['Keywords'].apply(lambda x: analyzer.polarity_scores(x))


    # In[338]:


    nytimes


    # # LOAD LOAD LOAD

    # In[11]:


    nytimes = pd.read_pickle("../nytimes_copy/nytimes_spacy.pickle")
    nytimes.head()


    # In[9]:


    dataframe = pd.read_csv("../nytimes_copy/master_df_R2.csv")


    # In[10]:


    dataframe['Keywords'].apply(lambda x: analyzer.polarity_scores(x))


    # In[25]:


    to_add = dataframe[['Dominant_Topic', 'Topic_Perc_Contrib','Keywords']]


    # In[320]:


    ## MASTER DF


    # In[26]:


    all_news = pd.concat([nytimes, to_add], axis=1)


    # In[27]:


    topics = []
    nytimes['topic0'].apply(lambda x: topics.append(x))
    nytimes['topic1'].dropna().apply(lambda x: topics.append(x))
    topics = set(topics)
    len(topics)

    pprint(topics)


    # In[14]:


    random = all_news.sample(frac=0.1)
    random.head()
    random = random[['title','content','Keywords','topic0','topic1']]
    random.to_csv('sample.csv')


    # In[ ]:


    for index, row in all_news.iterrows():



    # In[149]:


    #test_topics = [['africa'],
    #['americas', 'canada'],
    #['arts', 'artsspecial', 'dance','design','style','theater',
    # 'theaterspecial'],
    #['asia',],
    #['australia',],
    #['automobiles'],
    #['awardsseason',],
    #['baseball','hockey','basketball','fashion','football','golf','ncaabasketball',
    # 'ncaafootball','olympics','soccer','sports', 'tennis'],
    #['books','review'],
    #['briefing','editor','insider','nytnow','opinion','sunday'], # general news; mixed bag
    #['business','dealbook','economy','money','smallbusiness'],
    #['climate','earth'],
    #['cycling',],
    #['dining','eat','live','magazine'],
    #['edlife','education'],
    #['elections'],
    #['europe'],
    #['family'],
    #['health','well'],
    #['international','world'],
    #['middleeast',],
    #['movies','media','music','podcasts',],
    #['nyregion'],
    #['obituaries'],
    #['personaltech','technology'],
    #['politics'],
    #['realestate'],
    #['science','space'],
    #['travel'],
    #['upshot'],
    #['us'],
    #['weddings'],
    #['television']]
    #

    # In[28]:
    def to_text(doc, strings=False):
    if not strings:
        return [token.text for token in doc if not token.is_stop]
    else:
        return u" ".join([token.text for token in doc if not token.is_stop])


    # In[29]:


    nytimes['clean_strings'] = nytimes['spacy_docs'].apply(lambda x: to_text(x))


    # In[13]:


    # doc_list for conversion because we didn't know about goldparse
    doc_list = []
    for index, row in tqdm_notebook(nytimes.iterrows()):
    new = []
    for word in iter(row['clean_strings']):
        new.append(word)
    doc_list.append(new)


    # Create Dictionary

    id2word = gensim.corpora.Dictionary(doc_list)

    # Create Corpus
    texts = doc_list

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]


    # In[57]:


    len(corpus)


    # In[ ]:


    # To reload them:
    # dictionary = corpora.Dictionary.load('dictionary.dict')
    # corpus = corpora.MmCorpus('corpus.mm')


    # In[ ]:


    # To reload them:
    # bow_dictionary = corpora.Dictionary.load('bow_dictionary.dict')
    # bow_corpus = corpora.MmCorpus('bow_corpus.mm')


    # In[62]:


    for doc in corpus:
    for ix, freq in doc:
        print(f"WORD: {id2word[ix]}\tFREQ: {freq}\n")   
        time.sleep(0.1)


    # # LDA Topic modeling

    # In[64]:


    # One model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


    # In[66]:


    # save
    with open('lda_model.pickle', 'wb') as f:
    pickle.dump(lda_model, f)


    # In[266]:


    # open 
    with open('lda_model.pickle', 'rb') as f:
    lda_model = pickle.load(f)


    # In[9]:


    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]


    # In[68]:


    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()


    # In[69]:


    pprint(f"PERPLEXITY: {lda_model.log_perplexity(corpus)}") # a measure of how good the model is.
    pprint(f"COHERENCE: {coherence_lda}")


    # In[70]:


    fiz=plt.figure(figsize=(15,30))
    for i in range(10):
    df=pd.DataFrame(lda_model.show_topic(i), columns=['term','prob']).set_index('term')
    #     df=df.sort_values('prob')

    plt.subplot(5,2,i+1)
    plt.title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, palette='Reds_d')
    plt.xlabel('salience')


    plt.show()


    # In[153]:


    # def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    #     """
    #     Compute c_v coherence for number of topics
    # 
    #     Parameters:
    #     ----------
    #     dictionary : Gensim dictionary
    #     corpus : Gensim corpus
    #     texts : List of input texts
    #     limit : Max num of topics
    # 
    #     Returns:
    #     -------
    #     model_list : List of LDA topic models
    #     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    #     """
    #     coherence_values = []
    #     model_list = []
    #     for num_topics in tqdm_notebook(range(start, limit, step)):
    #         model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
    #         model_list.append(model)
    #         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())
    # 
    #     return model_list, coherence_values

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=doc_list, start=2, limit=60, step=6)

    # Show graph
    limit=60; start=2; step=6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence values"), loc='best')
    plt.show()


    # In[158]:


    pprint(f"MODEL LIST: {model_list}")
    pprint(f"COHERENCE VALUES: {coherence_values}")


    # In[278]:


    # bigram and trigram models
    # take as argument a list of a list of words

    bigram = gensim.models.Phrases(doc_list, min_count=5, threshold=500)
    trigram = gensim.models.Phrases(bigram[doc_list], threshold=500)

    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)


    # In[ ]:


    limit=40; start=2; step=6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


    # # Sklearn model

    # In[24]:


    # word_list = list(chain(*doc_list))
    doc_list


    # In[15]:


    nytimes.head()


    # In[25]:


    def to_text(doc, strings=False):
    if not strings:
        return [token.text for token in doc if not token.is_stop]
    else:
        return u" ".join([token.text for token in doc if not token.is_stop])

    # sklearn version
    # docs need to be a list of lists of STRINGS
    list_of_str = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))

    # settings that you use for count vectorizer will go here
    #tfidf_vectorizer=TfidfVectorizer(use_idf=True)

    # just send in all your docs here
    #tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(list_of_str)


    # In[30]:


    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[3]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    df.sort_values(by=["tfidf"],ascending=False)


    # In[35]:


    def split_random(matrix, percent_train=70, percent_test=15):
    """
    Splits matrix data into randomly ordered sets 
    grouped by provided percentages.

    Usage:
    rows = 100
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = \
    split_random(matrix, percent_train=80, percent_test=10)

    percent_validation 10
    training (80, 2)
    testing (10, 2)
    validation (10, 2)

    Returns:
    - training_data: percentage_train e.g. 70%
    - testing_data: percent_test e.g. 15%
    - validation_data: reminder from 100% e.g. 15%
    Created by Uki D. Lucas on Feb. 4, 2017
    """

    percent_validation = 100 - percent_train - percent_test

    if percent_validation < 0:
        print("Make sure that the provided sum of " +         "training and testing percentages is equal, " +         "or less than 100%.")
        percent_validation = 0
    else:
        print("percent_validation", percent_validation)

    #print(matrix)  
    rows = matrix.shape[0]
    np.random.shuffle(matrix)

    end_training = int(rows*percent_train/100)    
    end_testing = end_training + int((rows * percent_test/100))

    training = matrix[:end_training]
    testing = matrix[end_training:end_testing]
    validation = matrix[end_testing:]
    return training, testing, validation

    # TEST:
    rows = 100
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = split_random(matrix, percent_train=80, percent_test=10) 

    print("training",training.shape)
    print("testing",testing.shape)
    print("validation",validation.shape)

    print(split_random.__doc__)


    # In[52]:


    n_samples = tfidf_vectorizer_vectors.shape[1]


    # In[58]:


    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.datasets import make_multilabel_classification

    # rng = np.random.RandomState(42)  # reproducible results with a fixed seed
    # indices = np.arange(n_samples)
    # rng.shuffle
    # x_shuffled = [indices]
    # y_shuffled = y[indices]
    # 
    # train = x_shuffled
    # test = y_shuffled

    # This produces a feature matrix of token counts, similar to what
    # CountVectorizer would produce on text.
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda_model = lda.fit(tfidf_vectorizer_vectors) 

    # get topics for some given samples:
    print(lda.transform(tfidf_vectorizer_vectors[-2:]))

    lda_output = lda.transform(tfidf_vectorizer_vectors)

    print(lda_output)  # Model attributes


    # In[59]:


    # pyLDAvis.enable_notebook()
    # panel = pyLDAvis.sklearn.prepare(lda, tfidf_vectorizer_vectors, tfidf_vectorizer, mds='tsne')
    # panel


    # In[61]:


    # Diagnostics

    # Log Likelihood: Higher the better
    print(f"Log Likelihood: {lda_model.score(tfidf_vectorizer_vectors)}")

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print(f"Perplexity: {lda_model.perplexity(tfidf_vectorizer_vectors)}")

    # See model parameters
    print(f"Model params: {lda_model.get_params()}")


    # # TF/IDF Modeling

    # In[75]:


    import textacy
    from textacy import tm
    from textacy.vsm import Vectorizer
    matrix = Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth')
    doc_term_matrix = matrix.fit_transform(list_of_str)


    # In[76]:


    model = textacy.tm.TopicModel('nmf', n_topics=20)
    model.fit(doc_term_matrix)


    # In[308]:


    doc_topic_matrix = model.transform(doc_term_matrix)
    for topic_idx, top_terms in model.top_topic_terms(vectorizer.get_feature_names(), topics=[0,1]):
    print('topic', topic_idx, ':', '   '.join(top_terms[))


    # In[89]:


    # compare


    # ## Non-negative Matrix Factorization

    # In[66]:


    tfidf = TfidfVectorizer()
    tfidf_output = tfidf.fit_transform(list_of_str)


    # In[ ]:


    # vectorizer = TfidfVectorizer(
    #  stop_words=custom_list,
    #  min_df=20,
    #  max_df=1000,
    #  lowercase=False,
    #  ngram_range=2)


    # In[98]:


    from sklearn.decomposition import NMF, LatentDirichletAllocation

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    tfidf = tfidf_vectorizer.fit_transform(list_of_str)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    tf = tf_vectorizer.fit_transform(list_of_str)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 20

    # Run NMF
    nmf = NMF(n_components=77, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    no_top_words = 10


    # In[107]:


    display_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(lda, tf_feature_names, no_top_words)


    # In[ ]:


    display_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(lda, tf_feature_names, no_top_words)


    # # Testing, training, and prediction

    # ## TTS

    # In[253]:


    # data = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))
    # train = data.sample(frac=0.8, random_state=77)
    # test = data.drop(train.index)


    # ## Create pipelines

    # In[237]:


    # data = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))


    # In[15]:


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


    # In[254]:


    train


    # In[32]:


    from sklearn.pipeline import make_pipeline
    # vectorizers = [('count', CountVectorizer()), 
    #                ('tfidf', TfidfVectorizer())]
    # classifiers = [('svd', TruncatedSVD(n_components=300)), 
    #                ('lda', LatentDirichletAllocation()),
    #                ('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))]
    # parameters = {
    #     'tfidf__binary': (True, False),
    #     'tfidf__norm': ('l1', 'l2'),
    #     'tfidf__ngram_range':(1,3),
    #     'lda__n_components': [10, 25, 30, 35, 40],
    #     'lda__learning_decay': [.5, .7, .9],
    # }
    # 
    # pipe = make_pipeline([('count', CountVectorizer()),
    #                      ('svd', TruncatedSVD(n_components=300)), 
    #                #('lda', LatentDirichletAllocation()),
    #                #('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))
    #                      ])
    # 
    # grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)


    # In[ ]:


    #vect = Pipeline([('cnt', CountVectorizer())])
    #pipe = Pipeline([
    #            ('tf', TfidfVectorizer()),
    #            ('svd', TruncatedSVD(n_components=300))])

    #tfidf.fit(train)

    #lda = Pipeline([('svd', TruncatedSVD(n_components=300),
    #                 ('scaler', StandardScaler()),
    #                 ('lda', LatentDirichletAllocation()),
    #                 ('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))
    #
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    # 
    # class SkippableTruncatedSVD(TruncatedSVD):
    # 
    #     # add the "skip" argument and keep the others as in the superclass
    #     def __init__(self,skip=False,n_components=2, algorithm="randomized", n_iter=5,
    #                  random_state=None, tol=0.):
    #         self.skip = skip
    #         super().__init__(n_components, algorithm, n_iter, random_state, tol)
    # 
    #     # execute if not being skipped
    #     def fit(self, X, y=None):
    #         if self.skip:
    #             return self
    #         else:
    #             return super().fit(X,y)
    # 
    #     # execute if not being skipped
    #     def fit_transform(self, X, y=None):
    #         if self.skip:
    #             return X
    #         else:
    #             return super().fit_transform(X,y)
    # 



    # from sklearn.pipeline import make_pipeline 
    # from sklearn.pipeline import make_union
    # 
    # pipe = Pipeline([('svd', TruncatedSVD(n_components=300)),
    #                      ('lda', LatentDirichletAllocation())])
    # 
    # union = FeatureUnion([('idf', TfidfVectorizer()), 
    #                                 ('pipe', pipe),
    #                               ])
    #('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))
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
    #'class__c2__lda__topic_word_prior':np.arange(0.001,0.004,0.001)
    #'class__c2__lda__doc_topic_prior':np.arange(0.1,0.5,0.1)
    #'vec__tfidf__binary': (True, False)


    # # Adding bigrams and trigrams

    # In[258]:


    doc_list
    #
    #
    ## In[261]:
    #
    #
    ## See trigram example
    #trigram_model
    #
    #
    ## In[ ]:
    #
    #
    #spacy_docs = []
    #
    #for doc in tqdm_notebook(nlp.pipe(docs)):
    #    doc = [token.text for token in doc]
    #    doc = u' '.join(doc)
    #    spacy_docs.append(nlp.make_doc(doc))
    #    
    #
    #
    ## In[ ]:
    #
    #
    #with open('preprocess.pickle', 'rb') as f:
    #    nlp = pickle.load(f)
    #    
    #nlp.add_pipe(lemmatizer,name='lemmatizer',after='ner')
    #nlp.pipe_names
    #
    ## pickle this nlp pipeline
    #with open('all_pipes.pickle', 'wb') as f:
    #    pickle.dump(nlp, f)
    #
    #
    ## # Define the property extensions
    ## 
    ## https://support.prodi.gy/t/how-to-incorporate-document-metadata/296/3
    #
    ## In[ ]:
    #
    #
    #from spacy.tokens import Token
    #from spacy.tokenizer import Tokenizer
    #
    ## load the nlp preprocessor
    #with open('all_pipes.pickle', 'rb') as f:
    #    nlp = pickle.load(f)
    #
    #
    ## In[ ]:
    #
    #
    #nyt = pd.concat([nyt, sdf], axis=1)
    #
    #
    ## In[ ]:
    #
    #
    #nyt.head()
    #
    #
    ## In[ ]:
    #
    #
    ## get position of the doc from the content
    #def get_topic(doc):
    #    index = list(np.where(nyt['spacy_docs']==doc)[0])
    #    topic = nyt.iloc[index]['topic0']
    #    return topic
    #
    #Doc.set_extension("has_topic", getter=get_topic, force=True)
    #
    #
    ## # Clustering
    ## 
    ## 1. Bag of words: TF/IDF 
    ## 2. Given articles with these features:....
    #
    ## In[ ]:
    #
    #
    ## save the LDA pipeline
    #with open('LDA_nlp.pickle', 'w') as f:
    #    pickle.dump(f)
    #
    #
    ## In[ ]:
    #
    #
    ## load the model
    #with open('LDA_model.pickle', 'rb') as f:
    # lda = pickle.loads(f.read())
    #
    #
    ## In[ ]:
    #
    #
    ## mutual information scores
    #
    #
    ## In[ ]:
    #
    #
    ##pmi_finder = BigramCollocationFinder.from_words(alltext)
    #
    #
    ## In[ ]:
    #
    #
    ##pmi_finder.apply_freq_filter(6)
    ##pmi_scored = pmi_finder.score_ngrams(bigram_measures.pmi)
    ##pmi_scored

__name__ == "__main__" 