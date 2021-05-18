## API Key: ESgm5FayriHef86TKGHPXt00P
## API Secret Key: PWI1Z4030fYaZz2cFALtgpBAHt9ljH9q7ObIijdX34KL0UwvlY
## API Bearer Token: AAAAAAAAAAAAAAAAAAAAAMurOwEAAAAA9KQ%2BRRb0Ocw1jIl2lbQiRCH2rm4%3Dg4sUWC5mt70OCa0f7cjPecNmnBzYCKK7WUN6nE6y1Nv4vIMG6z
###
### Data Gathering/Cleaning
import pandas as pd

#getting data from the csv file with all the tweets and other data
tweet_data = pd.read_csv('nfl_tweets_hydrate.csv')
tweet_ids = tweet_data['id']
tweets = tweet_data['text']
id_to_tweet = {}

for i in range(len(tweets)):
    id_to_tweet[tweet_ids[i]] = tweets[i]

raw_data = pd.read_csv('tweets.nfl.2012.postgame.csv', header = None)
ids = raw_data[0]
team = raw_data[2]
opponent = raw_data[3]
team_score = raw_data[6]
opp_score = raw_data[7]
tweet_list = []
outcome_list = []

#labeling data with outcome, 1 or 0
for i in range(len(ids)):
    try:
        tweet = id_to_tweet[ids[i]]
        if team_score[i] > opp_score[i]:
            outcome = 1
        else:
            outcome = 0
        tweet_list.append(tweet)
        outcome_list.append(outcome)
    except KeyError:
        pass

#export only tweet and outcome to csv to use in code
our_data = pd.DataFrame({'Tweet':tweet_list,'Outcome':outcome_list})
our_data.to_csv('ThankGod.csv',index = None, header = ['Tweet','Outcome'], sep='\t')