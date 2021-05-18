import csv
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()


win_sentiment = []
lose_sentiment = []
spread_win_sentiment = []
spread_lose_sentiment = []
win_win = []
win_lose = []
lose_win = []
lose_lose = []
score_all = []
outcome_all = []

csvfile = open('Spread_Outcomes.csv', newline='')
tweetReader = csv.reader(csvfile, delimiter=',', quotechar='"')
next(tweetReader)

for num, tweet in enumerate(tweetReader):
    ## tweet[0] = tweet
    ## tweet[1] = win/lose
    ## tweet[2] = win/lose w/ spread
    try:
        text = " ".join(tweet_tokenizer.tokenize(tweet[0]))#tweet[0] #tweet_tokenizer.tokenize(tweet[0])
        #print(type(text))

        win_or_lose = int(tweet[1])
        #print(win_or_lose)
        win_or_lose_spread = int(tweet[2])
        #if win_or_lose_spread != win_or_lose:
        #    print(good)
        #blob = TextBlob(text)

        ### Compute Sentiment Score
        score = sid.polarity_scores(text)['compound']

        ## totals for fitting
        score_all.append(score)
        outcome_all.append(win_or_lose)

        if win_or_lose == 1:
            win_sentiment.append(score)
        else: 
            lose_sentiment.append(score)
        if win_or_lose_spread == 1:
            spread_win_sentiment.append(score)
        else: 
            spread_lose_sentiment.append(score)

        ## outcome_outcome analysis
        if win_or_lose and win_or_lose_spread:
            win_win.append(score)
        elif win_or_lose and not win_or_lose_spread:
            win_lose.append(score)
        elif not win_or_lose and not win_or_lose_spread:
            lose_lose.append(score)
        else:
            lose_win.append(score)

        #print(text, sid.polarity_scores(text)['compound'])
    except IndexError:
        # figure out what to do with wrong csv
        pass
        
print("Win Sentiment Average:", np.mean(win_sentiment), "STD:", np.std(win_sentiment))
print("Lose Sentiment Average:", np.mean(lose_sentiment), "STD:", np.std(lose_sentiment))
print("Win Spread Sentiment Average:", np.mean(spread_win_sentiment), "STD:", np.std(spread_win_sentiment))
print("Lose Spread Sentiment Average:", np.mean(spread_lose_sentiment), "STD:", np.std(spread_lose_sentiment))
#print(win_sentiment)
print("-------------------------------")
print("Win the game, beat the spread:", np.mean(win_win), "STD:", np.std(win_win), "N:", len(win_win))
print("Win the game, lose the spread:", np.mean(win_lose), "STD:", np.std(win_lose), "N:", len(win_lose))
print("Lose the game, beat the spread:", np.mean(lose_win), "STD:", np.std(lose_win), "N:", len(lose_win))
print("Lose the game, lose the spread:", np.mean(lose_lose), "STD:", np.std(lose_lose), "N:", len(lose_lose))



## Make graphs of the histograms
n, bins, patches = plt.hist(lose_sentiment, 20, facecolor='blue', alpha=0.5)
#plt.show()


## Model fitting

X_train, X_test, y_train, y_test = train_test_split(np.array(score_all), np.array(outcome_all), test_size=0.2, random_state=100)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

svm = LinearSVC().fit(X_train, y_train)

predicted = svm.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))
print(svm.intercept_)

## Ideas:
# RNN Classification for Tweets -> Won/Lost game
# Sentiment Analysis Classification -> Won/Lost game : Can present this
# 
#
#
#
#
#
## use textblob


## Some examples from Vader sentiment analysis
# RT @CrimsonTide85: S/O to RG3 and The RedSkins on beating them Saints #Loyalfan #REDSKINNATION #FirstOfMany -0.4588
# It wasn't picture perfect, however, #Eagles got job done today against #Browns. Better to win ugly than not all. Go Birds! #FlyEaglesFly 0.1764
# I'm not a #Redskins fan but ya gotta love #RGIII 0.7445
# #JETS!!!!! :D 0.6671
# RT @ShesIn_hEVIN: Damn heard my boys ain't do to good..they gone be iite tho #RamNation -0.4019


