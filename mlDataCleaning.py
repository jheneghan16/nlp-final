import pandas as pd

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
spread = raw_data[8]
tweet_list = []
outcome_list = []
spread_outcome_list = []

#labeling data with outcome, 1 or 0
for i in range(len(ids)):
    try:
        tweet = id_to_tweet[ids[i]]
        if team_score[i] > opp_score[i]:
            outcome = 1
        else:
            outcome = 0
        if team_score[i] + spread[i] > opp_score[i]:
            spread_outcome = 1
        else:
            spread_outcome = 0
        #if outcome != spread_outcome:
        #    print("good")
        tweet_list.append(tweet)
        outcome_list.append(outcome)
        spread_outcome_list.append(spread_outcome)
    except KeyError:
        pass

#export only tweet and outcome to csv to use in code
our_data = pd.DataFrame({'Outcome':outcome_list,'Tweet':tweet_list})
our_data.to_csv('MLDataset.csv',index = None, header=False, sep='\t')