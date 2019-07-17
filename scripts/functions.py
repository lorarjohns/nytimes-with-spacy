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

for word in stop_list:
    print(nlp.vocab[word].is_stop)

def lemmatizer(doc):
    doc = [token.lemma_ for token in doc if not token.lemma_ == '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)

nlp.add_pipe(lemmatizer, 'lemmatizer', after='tagger')

def to_text(doc, strings=False):
    if not strings:
        return [token.text for token in doc if not token.is_stop]
    else:
        return u" ".join([token.text for token in doc if not token.is_stop])

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm_notebook(range(start, limit, step)):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



class SkippableTruncatedSVD(TruncatedSVD):

    # add the "skip" argument and keep the others as in the superclass
    def __init__(self,skip=False,n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.skip = skip
        super().__init__(n_components, algorithm, n_iter, random_state, tol)

    # execute if not being skipped
    def fit(self, X, y=None):
        if self.skip:
            return self
        else:
            return super().fit(X,y)

    # execute if not being skipped
    def fit_transform(self, X, y=None):
        if self.skip:
            return X
        else:
            return super().fit_transform(X,y)

def get_topic(doc):
    index = list(np.where(nyt['spacy_docs']==doc)[0])
    topic = nyt.iloc[index]['topic0']
    return topic

Doc.set_extension("has_topic", getter=get_topic, force=True)