#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from gensim.models import ldamodel
from gensim.models import Word2Vec
import gensim.corpora
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pickle


# In[2]:


data = pd.read_csv('./clean_jd_data.csv')


# In[20]:


data.pseg_words = data.pseg_words.apply(literal_eval)


# In[64]:


attrs = list(
    set([attr for ls in data.pseg_words for elem in ls for wd, attr in elem]) 
    - set(['a', 'd', 'i', 'l'])
)


# In[75]:


data['words'] = [[wd for elem in ls for wd, attr in elem if attr in attrs and len(wd) > 1] for ls in data.pseg_words]
data_text = data[['words']]
train_headlines = [value[0] for value in data_text.values]


# # LDA implement

# In[ ]:


num_topics = 10


# In[79]:


id2word = gensim.corpora.Dictionary(train_headlines)
corpus = [id2word.doc2bow(text) for text in train_headlines]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)


# In[80]:


def get_lda_topics(model, num_topics, topn):
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn=topn)
        word_dict['Topic' + '{:02d}'.format(i+1)] = [word[0] for word in words]
    return pd.DataFrame(word_dict)


# In[81]:


get_lda_topics(lda, num_topics, 30)


# # NMF implement

# In[84]:


sentences = [' '.join(text) for text in train_headlines]


# In[85]:


# The CountVectorizer module return a matrix of size(Documents X Features), where the value of 
# a cell is going to be the number of times of the feature (word) appear in that document.
vectorizer = CountVectorizer(analyzer='word', max_features=5000)
x_counts = vectorizer.fit_transform(sentences)


# In[87]:


# Set a TFIDF transformer, transform the counts with the model and normalize the values
transformer = TfidfTransformer(smooth_idf=False)
x_tfidf = transformer.fit_transform(x_counts)
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


# In[90]:


model = NMF(n_components=num_topics, init='nndsvd')
model.fit(xtfidf_norm)


# In[102]:


def get_nmf_topics(model, num_topics, topn):
    feat_names = vectorizer.get_feature_names()
    word_dict = {}
    
    for i in range(num_topics):
        words_ids = model.components_[i].argsort()[:-topn-1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic ' + '{:02d}'.format(i+1)] = words
    return pd.DataFrame(word_dict)


# In[139]:


num_topics = 5
get_nmf_topics(model, num_topics, 30)


# In[ ]:




