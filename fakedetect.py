# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:19:19 2017

Author: Ngo Duy Vu
Project:DeeepSense customer review
Classify Trustful Fake review
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from gensim import matutils,corpora
from sklearn.externals import joblib

def extract_tokens(df):
    """ Tokenize a sentence into words, add tag for each word  """
    review_tokenized = []
    lmt = WordNetLemmatizer()
    for datapoint in df:
        tokenize_words = word_tokenize(datapoint.lower(),language='english')
        pos_word = pos_tag(tokenize_words)
        tokenize_words = ["_".join([lmt.lemmatize(i[0]),i[1]]) for i in pos_word if (i[0] not in stopwords.words("english") and len(i[0]) > 2)]
        review_tokenized.append(tokenize_words)
    return review_tokenized


def vectorize_review(df, dictionary):
    """ Convert each tokenized sentence into vectors """
    corpus = [dictionary.doc2bow(text) for text in df]
    corpus = matutils.corpus2csc(corpus, num_terms=len(dictionary.token2id))
    corpus = corpus.transpose()
    return corpus

def opinion_spam_classify(all_documents):
    """ Classify deceptive/truthful in each review """
    deceptive = []
    truthful = []
    dictionary = corpora.Dictionary.load('Secret_Project\\dictionary\\deerwester_2017-09-07-16-42-06.dict')
    nn_clf = joblib.load('Secret_Project\\neural_network\\nn_model_2017-09-07-16-42-06.pkl')
    tokens_documents = extract_tokens(all_documents)
    vector_documents = vectorize_review(tokens_documents, dictionary)
    for review in vector_documents: 
        result = nn_clf.predict_proba(review)
        deceptive.append(result[0][0])
        truthful.append(result[0][1])
    return [deceptive, truthful]
