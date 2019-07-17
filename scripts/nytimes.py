nytimes = pd.read_csv('nytimes.csv')
nytimes.columns

nytimes.reset_index(inplace=True, drop=True, col_level=0)
nytimes.drop(['Unnamed: 0', 'id', 'stems'], axis=1, inplace=True)
topic_re = r"(?<=\d{4}\/\d{2}\/\d{2}\/)(\w+\-?\w?\/?\w+\/)"
topics = nytimes['url'].str.extract(topic_re, expand=True)
topics.dropna(inplace=True)
nytimes.drop(index=3886, inplace=True)
topics[0] = topics[0].map(lambda x: x.replace('/', ' '))
pattern = re.compile("(\w+) (\w+)?")
add_topics = topics[0].str.extract("(\w+) (\w+)?", expand=True)
nytimes = pd.concat([nytimes, add_topics], axis=1)
# see date range
date = pd.to_datetime(nytimes['date'])
date = pd.DataFrame(date)
cols = {0 : "topic0", 1 : "topic1"}
nytimes.rename(columns=cols, inplace=True)

topics = [x for x in [list(nytimes['topic0'].unique()), list(nytimes['topic1'].unique())]]

topicsdf = pd.concat([nytimes['topic0'],nytimes['topic1']], axis=0)
topicsdf.fillna("")
topicsdf = pd.DataFrame(topicsdf)
topicsdf.columns = ["all_topics"]

topicsdf.dropna(inplace=True)

fig, ax = plt.subplots(1,2)
sns.set(rc={'figure.figsize':(30,25)})
sns.countplot(nytimes['topic0'], ax=ax[0])
ax_twin = ax.twinx()

sns.countplot(nytimes['topic1'], ax=ax[1])
plt.xticks(rotation=75, horizontalalignment='right')
fig.show()

# TRYING IT OUT
nlp = spacy.load('en_core_web_lg') # loading the language model 
doc = nlp(nytimes['content'][0])
sentences = list(doc.sents) 

displacy.serve(sentences, style="dep")
displacy.render(doc, style="ent")
for token in doc: 
print(f"TOKEN: {token}\t\t\t POS:{token.pos_}")

nlp = spacy.load('en_core_web_lg')
#or doc in tqdm_notebook(nlp.pipe(docs)):
#   doc = [token.text for token in doc]
#   doc = u' '.join(doc)
#   spacy_docs.append(nlp.make_doc(doc))

with open('all_pipes.pickle', 'rb') as f:
    nlp = pickle.load(f)  

stopwords = spacy.lang.en.STOP_WORDS
stop_list = ["Mr.","Mrs.","Ms.","say","WASHINGTON","'s"]
nlp.Defaults.stop_words.update(stop_list)

def cleaner(doc):
    doc = [token.text for token in doc if (not token.is_stop and not token.is_punct)]
    doc = u' '.join(doc)
    return nlp.make_doc(doc)

nlp.add_pipe(cleaner, name="cleaner", first="true")

doc = nlp(nytimes['content'][0])
with open('preprocess.pickle', 'wb') as f:
    pickle.dump(nlp, f)

    for token in doc:
    print(f"TOKEN: {token}\t\tPOS: {token.pos_}")

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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

nytimes.drop(['sentiment'], axis=1, inplace=True)

nytimes['sentiment'] = dataframe['Keywords'].apply(lambda x: analyzer.polarity_scores(x))

dataframe = pd.read_csv("master_df_R2.csv")

to_add = dataframe[['Dominant_Topic', 'Topic_Perc_Contrib','Keywords']]

all_news = pd.concat([nytimes, to_add], axis=1)

topics = []
nytimes['topic0'].apply(lambda x: topics.append(x))
nytimes['topic1'].dropna().apply(lambda x: topics.append(x))
topics = set(topics)
len(topics)

pprint(topics)

random = all_news.sample(n=20)
random.head()
random = random[['title','content','Keywords','topic0','topic1']]
random.to_csv('sample.csv')

test_topics = [['africa'],
['americas', 'canada'],
['arts', 'artsspecial', 'dance','design','style','theater',
 'theaterspecial'],
['asia',],
['australia',],
['automobiles'],
['awardsseason',],
['baseball','hockey','basketball','fashion','football','golf','ncaabasketball',
 'ncaafootball','olympics','soccer','sports', 'tennis'],
['books','review'],
['briefing','editor','insider','nytnow','opinion','sunday'], # general news; mixed bag
['business','dealbook','economy','money','smallbusiness'],
['climate','earth'],
['cycling',],
['dining','eat','live','magazine'],
['edlife','education'],
['elections'],
['europe'],
['family'],
['health','well'],
['international','world'],
['middleeast',],
['movies','media','music','podcasts',],
['nyregion'],
['obituaries'],
['personaltech','technology'],
['politics'],
['realestate'],
['science','space'],
['travel'],
['upshot'],
['us'],
['weddings'],
['television']]

nytimes['clean_strings'] = nytimes['spacy_docs'].apply(lambda x: to_text(x))

# doc_list will be our all-important source of things
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

for doc in corpus:
    for ix, freq in doc:
        print(f"WORD: {id2word[ix]}\tFREQ: {freq}\n")   
        time.sleep(0.1)


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

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

pprint(f"PERPLEXITY: {lda_model.log_perplexity(corpus)}") # a measure of how good the model is.
pprint(f"COHERENCE: {coherence_lda}")

fiz=plt.figure(figsize=(15,30))
for i in range(10):
    df=pd.DataFrame(lda_model.show_topic(i), columns=['term','prob']).set_index('term')
#     df=df.sort_values('prob')
    
    plt.subplot(5,2,i+1)
    plt.title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, palette='Reds_d')
    plt.xlabel('salience')
    

plt.show()


update_stop_words(stop_list)

model_list, coherence_values = compute_coherence_values(dictionary=id2word, 
corpus=corpus, texts=doc_list, start=2, limit=60, step=6)

# Show graph
limit=60; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence values"), loc='best')
plt.show()


pprint(f"MODEL LIST: {model_list}")
pprint(f"COHERENCE VALUES: {coherence_values}")

# bigram and trigram models
# take as argument a list of a list of words

bigram = gensim.models.Phrases(doc_list, min_count=5, threshold=500)
trigram = gensim.models.Phrases(bigram[doc_list], threshold=500)

bigram_model = gensim.models.phrases.Phraser(bigram)
trigram_model = gensim.models.phrases.Phraser(trigram)

limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# sklearn version

# docs need to be a list of lists of STRINGS
list_of_str = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))
vectorizer = CountVectorizer(min_df=10)

data_vectorized = vectorizer.fit_transform(list_of_str)

density = data_vectorized.todense()

vectorizer.vocabulary_

#Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((density > 0).sum()/density.size)*100, "%")

lda_model_sk = LatentDirichletAllocation()
lda_output = lda_model_sk.fit_transform(data_vectorized)

print(lda_model_sk)  # Model attributes

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model_sk, data_vectorized, vectorizer, mds='tsne')
panel

# Diagnostics

# Log Likelihood: Higher the better
pprint(f"Log Likelihood: {lda_model_sk.score(data_vectorized)}")

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
pprint(f"Perplexity: {lda_model_sk.perplexity(data_vectorized)}")

# See model parameters
pprint(f"Model params: {lda_model_sk.get_params()}")

matrix = Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth')
doc_term_matrix = matrix.fit_transform(list_of_str)

model = textacy.tm.TopicModel('nmf', n_topics=20)
model.fit(doc_term_matrix)   

doc_topic_matrix = model.transform(doc_term_matrix)
for topic_idx, top_terms in model.top_topic_terms(vectorizer.get_feature_names(), topics=[0,1]):
    print('topic', topic_idx, ':', '   '.join(top_terms[]))

tfidf = TfidfVectorizer()
tfidf_output = tfidf.fit_transform(list_of_str)  

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

display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

data = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))
train = data.sample(frac=0.8, random_state=77)
test = data.drop(train.index)

data = nytimes['spacy_docs'].apply(lambda x: to_text(x, strings=True))

vectorizers = [('count', CountVectorizer()), 
               ('tfidf', TfidfVectorizer())]
classifiers = [('svd', TruncatedSVD(n_components=300)), 
               ('lda', LatentDirichletAllocation()),
               ('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))]
parameters = {
    'tfidf__binary': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'tfidf__ngram_range':(1,3),
    'lda__n_components': [10, 25, 30, 35, 40],
    'lda__learning_decay': [.5, .7, .9],
}

pipe = make_pipeline([('count', CountVectorizer()),
                     ('svd', TruncatedSVD(n_components=300)), 
               #('lda', LatentDirichletAllocation()),
               #('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1))
                     ])

grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)

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



union = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([ 
            ('tfidf', TfidfVectorizer(max_df=0.25)),
            ('svd', SkippableTruncatedSVD(n_components=300)),
                ])),
        ('words', Pipeline([
            ('scaler', StandardScaler()),
        ])),
            ])),
    ('lda', LatentDirichletAllocation()),
    ('xgb', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
])

parameters = {
    'features__tf__binary': (True, False),
    'features__tf__norm': ('l1', 'l2'),
    'features__tf__ngram_range':(1,3),
    'features__svd__skip':[True,False],
    'union__lda__n_components': [10, 25, 30, 35, 40],
    'union__lda__learning_decay': [.5, .7, .9],
}

gridsearch = GridSearchCV(union, param_grid=parameters, cv=5,
                               n_jobs=-1, verbose=1)
gridsearch.fit(train)

spacy_docs = []

class MyClass:
    def __init__(self, name):
        self.name = name

def pickle_MyClass(obj):
    assert type(obj) is MyClass
    return program.MyClass, (obj.name,)

copyreg.pickle(MyClass, pickle_MyClass)

if __name__ == '__main__':
    o = MyClass('test')
    with open('out.pkl', 'wb') as f:
        pickle.dump(o, f)


                            

    
# load the nlp preprocessor
