# -*- coding: utf-8 -*-
"""
Author: Ngo Duy Vu
Project:DeeepSense customer review

Read data and output sentiment
"""
# Import necessary library
import re
import json
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from autocorrect import spell
from replacers import RepeatReplacer, RegexpReplacer
from tfidf import find_idf
from fakedetect import opinion_spam_classify


## Download necessary package of 
#nltk.download('popular')

def loading_data(file_name):
    """ Loading data from JSON format """
    with open(file_name, 'r') as file:
        data = (json.loads(line) for i, line in enumerate(file.readlines()))
    return data

        

# Classify model    
def score_to_label(score):
    """ Classify the sentence according
    to the compound porlarity """ 
    if score <= -0.5:
        return 0   # Negative
    if(score > -0.5) & (score < -0.1):
        return 1
    if(score >= -0.1) & (score <= 0.1):
        return 2
    if(score > 0.1) & (score < 0.5):
        return 3
    if score >= 0.5:
        return 4

# Find the polarity of the sentiment    
def find_sentement_score(tokenized):
    """ Find the sentiment component of the sentence """
    compound_sum = 0
    sentence_count = 0
    sid = SentimentIntensityAnalyzer()
    if type(tokenized) == str:
        ss = sid.polarity_scores(tokenized)
        sentiment_score = ss['compound']
        return sentiment_score 
    else:
        for sentence in tokenized:
            ss = sid.polarity_scores(sentence)
            compound_sum+=ss['compound']
            sentence_count+=1
    sentiment_score = (compound_sum/sentence_count)
    return sentiment_score

# Cleaning the review by checking spell, delete punctuation
def cleaning_text(sentence):
    """ Cleaning the sentence by checking 
    the spell for each word, replace bad 
    character """
    regex = RegexpReplacer()
    repeat = RepeatReplacer()
    sentence = sentence.lower()
    words = [repeat.replace(i) for i in sentence.split(" ")]
    words = [regex.replace(i) for i in words]
    sentence = ' '.join(words)
    words = [spell(i) for i in sentence.split(" ")]
    sentence = ' '.join(words)
    sentence = re.sub('<[^>]*>','',sentence)
    smileys = re.findall('((?::|;|=)(?:-?)(?:[D|d|)|(|P|p|/|x|X]))',sentence)
    sentence = re.sub('[\W]+',' ',sentence)
    sentence += ' '.join(smileys).replace('-','')
    return sentence


# Saving the data after process
def save_to_csv(save_name, data):
    """ Saving the produced data """
    data.to_csv(save_name, sep=',', encoding='utf-8')

def sentiment_analysis(dataset):
    """ Find the sentiment and polarity of the reivews """
    index = 0
    classied_df = pd.DataFrame(columns=['Review Text', 'Length', 'Sentiment', 'Polarity'])
    for review in dataset:
        review_len = len(review)
        data_clean = cleaning_text(review)
        polarity = find_sentement_score(data_clean)
        data_label = score_to_label(polarity)
        classied_df.loc[index] = [review, review_len, data_label, abs(polarity)]
        index+=1
    return classied_df
    

def main():
    #df = pd.read_csv("Data Library\Small_data.csv", error_bad_lines=False)
    df = pd.read_csv("Small_data.csv", error_bad_lines=False)
    new_df = sentiment_analysis(df['reviewText'])
    similar, keyword = find_idf(df['reviewText'])
    deceptive, truthful = opinion_spam_classify(df['reviewText'])
    new_df['KeyWord'] = keyword
    new_df['Similarity'] = similar
    new_df['%Deceptive'] = deceptive
    new_df['%Truthful'] = truthful
    save_to_csv('file_save.csv', new_df)


if __name__ == "__main__":
    main()