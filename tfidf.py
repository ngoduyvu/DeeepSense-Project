# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:06:33 2017
Author: Ngo Duy Vu
Project:DeeepSense customer review
    
TF-IDF and cosine distance to 
find the similar document and keywords
"""
import math
import operator
import string
from nltk.corpus import stopwords
from autocorrect import spell
from replacers import RepeatReplacer, RegexpReplacer

def cleaning_text(sentence):
    """ Cleaning text for tfidf """
    regex = RegexpReplacer()
    repeat = RepeatReplacer()
    sentence = sentence.lower()
    words = [repeat.replace(i) for i in sentence.split(" ")]
    words = [regex.replace(i) for i in words]
    sentence = ' '.join(words)
    sentence = [spell(i) for i in sentence.split(" ")]
    sentence = ' '.join(words)
    sentence = [s for s in sentence if s not in string.punctuation]
    return ''.join(sentence)

def unique_word(all_document):
    """ Create a list of all unique words
    all the whole document """
    tokenize = lambda doc: doc.split(" ")       # Tokenize sentence to words
    tokenized_documents = [tokenize(d) for d in all_document]
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    return [all_tokens_set, tokenized_documents]

def term_frequency(term, tokenized_document):
    """ Count many time word appear in a document """
    frequency = tokenized_document.count(term)
    return frequency

def sublinear_term_frequency(term, tokenized_document):
    """ Normalize the sentence to reduce 
    the length affect on the frequency of 
    words """
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return (1+ math.log(count))

def inverse_document_frequencies(list_words, tokenized_documents):
    """ The rare words usually carry significant meaning 
    for a sentence while popular words is not, inversing 
    the frequencies of rare and popular words to ensure 
    words have correct weight values """
    idf_values = {}
    for token in list_words:
        contains_token =map(lambda doc: token in doc, tokenized_documents)
        idf_values[token] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
        return idf_values
    
def tfidf(tokenized_documents, idf):
    """ Convert a document into a vector of word
    with each value component of the vector is 
    the weight of words """
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents


def cosine_similarity(vector1, vector2):
    """ Measure the distance of 2 vector, the vectors
    similar to each other tend to be near in space """
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return (dot_product/magnitude)
    
def idf_keyword(tokenized_documents, idf_values):
    """ Find the keyword base on the weight IDF 
    of word, the function the five word which have
    largest weight value """
    counter = 0;
    dict_keyword = {}
    list_keyword = []
    sentence_keyword = []
    stops = set(stopwords.words('english'))
    for review in tokenized_documents:
        gen = (i for i in review if i not in stops)
        for word in gen:
            if word in idf_values:
                dict_keyword[word] = idf_values[word]
            else:
                dict_keyword[word] = 7
        sort_dict = sorted(dict_keyword.items(), key=operator.itemgetter(1), reverse=True)[:5]
        for x in sort_dict:
            list_keyword.append(x[0])
        sentence_keyword.append(set(list_keyword))
        list_keyword = []
        counter+=1
        dict_keyword.clear()
    return sentence_keyword

def similar_score(tokenized_documents, idf_values):
    """ Function range the document based on similarity
    with others, the more different the document is
    higher score it get """
    tfidf_comparisons = []
    tfidf_representation = tfidf(tokenized_documents, idf_values)
    for index1, review1 in enumerate(tfidf_representation):
        score = 0
        for index2, review2 in enumerate(tfidf_representation):
            if index1 != index2:
                similarity = cosine_similarity(review1, review2) 
                if similarity >= 0.9:
                    score = score + 1
                elif (similarity < 0.9) & (similarity >= 0.7):
                    score = score + 2
                elif (similarity < 0.7) & (similarity >= 0.5):
                    score = score + 5
                elif (similarity < 0.5) & (similarity >= 0.2):
                    score = score + 10
                elif (similarity < 0.2):
                    score = score + 20        
        tfidf_comparisons.append(score)
    return tfidf_comparisons


def find_idf(all_documents):
    """ Find the keyword and similarity score of
    the document """
    all_reviews = []
    for review in all_documents:
        sentence = cleaning_text(review)
        all_reviews.append(sentence)
    list_words, tokenized_documents = unique_word(all_reviews)
    idf_values = inverse_document_frequencies(list_words, tokenized_documents)
    similar = similar_score(tokenized_documents, idf_values)
    keyword = idf_keyword(tokenized_documents, idf_values)
    return [similar, keyword]
        
        
    
