# -*- coding: utf-8 -*-
"""
Author: Ngo Duy Vu
Project:DeeepSense customer review

Read data and output sentiment
"""
# Import necessary library
import re
import pandas as pd
from csv import DictReader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from autocorrect import spell
from replacers import RepeatReplacer, RegexpReplacer
from tfidf import find_idf
from fakedetect import opinion_spam_classify


# Download necessary package of 
##nltk.download('popular')

def column_to_list(fileName, columnName):
    """ Loading the columns of the file and store it into list """
    with open(fileName) as f:
        listData = [column[columnName] for column in DictReader(f)]
    return listData

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
    #words = [spell(i) for i in sentence.split(" ")]
    #sentence = ' '.join(words)
    # Find all emoji symbol in the text
    sentence = re.sub('<[^>]*>','',sentence)
    smileys = re.findall('((?::|;|=)(?:-?)(?:[D|d|)|(|P|p|/|x|X]))',sentence)
    sentence = re.sub('[\W]+',' ',sentence)
    sentence += ' '.join(smileys).replace('-','')
    return sentence


# Saving the data after process
def save_to_csv(save_name, data, batchSize, counter):
    """ Saving the produced data """
    data = data[['Asin', 'Overall', 'ReviewText', 'KeyWord', 'Length', 'SentimentLabel', 'Polarity',
                'Similarity', 'Truthful', 'Deceptive', 'Helpfull', 'Unhelpfull']]
    #data.to_csv(save_name, sep=',', encoding='utf-8')
    if counter == 0:
        data.to_csv(save_name, index=False, header=True, chunksize=batchSize)
    else:
        data.to_csv(save_name, index=False, header=False, mode='a', chunksize=batchSize)

def sentiment_analysis(dataset):
    """ Find the sentiment and polarity of the reivews """
    reviewLen = []
    polarity = []
    sentimentLabel = []
    for review in dataset:
        reviewLen.append(len(review))
        setimentScore = find_sentement_score(review)
        polarity.append(setimentScore)
        sentimentLabel.append(score_to_label(setimentScore))
    return [reviewLen, sentimentLabel, polarity]
   
def lists_to_df(reviewLen, dataLabel, polarity, similar, keyword, deception, truthful):
    """ Convert all the result from other function from list to DataFrame """
    df = pd.DataFrame()
    df = df.assign(Length = reviewLen, SentimentLabel = dataLabel, Polarity = polarity)
    df = df.assign(Similarity = similar, KeyWord = keyword)
    df = df.assign(Truthful = truthful, Deceptive = deception)
    return df

def main():
    newDf = pd.DataFrame()
    counter = 0     # Check the first time function take in df to store
    batchSize = 100
    # Read 100 lines each time from the data source
    for df in pd.read_csv('Big_data.csv', chunksize=batchSize, iterator=True):
        reviewText = [cleaning_text(text) for text in df['reviewText']]
        reviewLen, dataLabel, polarity = sentiment_analysis(reviewText)
        similar, keyword = find_idf(reviewText)
        deceptive, truthful = opinion_spam_classify(reviewText)
        newDf = lists_to_df(reviewLen, dataLabel, polarity, similar, keyword, deceptive, truthful)
        newDf = newDf.set_index(df.index)   # Matching the index everytime reading 
        # Copy the columns of the original dataframe to the new
        newDf[['Asin', 'ReviewText', 'Overall', 'Helpfull', 'Unhelpfull']] = df[["asin", 
            "reviewText", "overall", "Helpfull", "Unhelpfull"]]
        save_to_csv('Big_file.csv', newDf, batchSize, counter)
        counter+=1
    del(newDf)

if __name__ == "__main__":
    main()
