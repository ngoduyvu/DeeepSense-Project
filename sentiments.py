# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:00:36 2017

@author: s4341237
"""
# Import necessary library
import re
import nltk
import string
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from autocorrect import spell
from replacers import RepeatReplacer, RegexpReplacer

## Download necessary package of 
#nltk.download('popular')

# Reading data file
def loading_data(file_name):
    with open(file_name, 'r') as file:
        data = (json.loads(line) for i, line in enumerate(file.readlines()))
    return data

        

# Classify model    
def score_to_label(score):
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
    compound_sum = 0
    sentence_count = 0
    sid = SentimentIntensityAnalyzer()
    if type(tokenized) == str:
        ss = sid.polarity_scores(tokenized)
        sentiment_score = ss['compound']
        return score_to_label(sentiment_score)
    else:
        for sentence in tokenized:
            ss = sid.polarity_scores(sentence)
            compound_sum+=ss['compound']
            sentence_count+=1
    sentiment_score = (compound_sum/sentence_count)
    return score_to_label(sentiment_score) 

# Cleaning the review by checking spell, delete punctuation
def cleaning_text(sentence):
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
    sentence = re.sub('[\W]+',' ',sentence.lower())
    sentence += ' '.join(smileys).replace('-','')
    return sentence


# Saving the data after process
def save_to_csv(save_name, data):
    data.to_csv(save_name, sep=',', encoding='utf-8')

def main():
    index = 0
    dataset = loading_data("small_data.json")
    df = pd.DataFrame([index for index in dataset])
    classied_df = pd.DataFrame(columns=['Review Text', 'Sentiment'])
    for review in df['reviewText']:
        data_clean = cleaning_text(review)
        data_label = find_sentement_score(data_clean)
        classied_df.loc[index] = [review, data_label]
        index+=1
    save_to_csv('test_save.csv', classied_df)
    


if __name__ == "__main__":
    main()