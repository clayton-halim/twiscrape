import json
import os
from datetime import datetime, timezone
import time

import numpy as np
from twython import Twython
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Twiscrape():
    MAX_TWEETS_PER_REQUEST = 100

    def __init__(self, app_key, app_secret):
        self.twitter = Twython(app_key, app_secret, oauth_version=2)
        access_token = self.twitter.obtain_access_token()
        self.twitter = Twython(app_key, access_token=access_token)

    def rate_limit(resource, option):
        """
        Returns a tuple in the order of how many lookup api calls remain and the time
        in UTC when the limit resets.

        twitter - Twython Client
        """

        rate_limits = client.get_application_rate_limit_status(resources=resource)
        option_limit = rate_limits['resources'][resource]['/{}}/{}'.format(
                            resource, option)]

        return (option_limit['remaining'], option_limit['reset'])

    def get_user(self, user):
        """Returns information of user's account in JSON format and save it as [user].json"""

        profile = self.twitter.show_user(screen_name=user)

        with open(os.path.join('users', 'profiles', user + '.json'), 'w') as fp:
            json.dump(profile, fp)

        return profile

    def get_user_tweets(self, user):
        """
        Returns a user's tweets in JSON format and stores it locally as [user].py
        
        NOTE: the Twitter API only allows a maximum of 3200 tweets to be scraped
        """

        since_id = 20 # the oldest tweet that you want
        oldest = 0 # oldest tweet id read so far
        timeline = []
        timeline_old = []

        # Check if user data already exists, if so read it in
        if os.path.isfile(os.path.join('users', 'data', user + '.json')):
            with open(os.path.join('users', 'data', user + '.json'), 'r') as fp:
                timeline_old = json.load(fp)

        if len(timeline_old) > 0:
            since_id = timeline_old[0]['id']

        remaining_get, reset_time = self.rate_limit('statuses', 'user_timeline')

        if remaining_get == 0:
            time.sleep(reset_time - time.time() + 1)
            remaining_get, reset_time = self.rate_limit('statuses', 'user_timeline')

        tweets = self.twitter.get_user_timeline(screen_name=user, include_rts=False,
                                                    count=200, trim_user=True, since_id=since_id)
        timeline.extend(tweets)
        remaining_get -= 1

        done = False    # check for more tweets

        if len(tweets) > 0:
            oldest = tweets[-1]['id'] - 1
        else:
            done = True

        # Check if there are anymore tweets to scrape
        while not done:
            if remaining_get == 0:
                time.sleep(reset_time - time.time() + 1)
                remaining_get, reset_time = rate_limit('statuses', 'user_timeline')

            tweets = self.twitter.get_user_timeline(screen_name=user, include_rts=False,
                                                      count=200, trim_user=True, max_id=oldest,
                                                    since_id=since_id)
            timeline.extend(tweets)
            remaining_get -= 1

            if len(tweets) > 0:
                oldest = tweets[-1]['id'] - 1
            else:
                done = True

        timeline.extend(timeline_old)

        with open(os.path.join('users', 'data', user + '.json'), 'w') as fp:
            json.dump(timeline, fp)

        return timeline

    def update_user_tweets(self, user, count=None):
        """
        Update the user's timeline stored locally
        
        count (int): specifies how many of the most recent tweets you want updated. None by default,
                        which means all tweets will be updated. 
        """
        
        timeline = []
        new_timeline = []

        if not os.path.isfile(os.path.join('users', 'data', user + '.json')):
            raise FileNotFoundError

        with open(os.path.join('users', 'data', user + '.json'), 'r') as fp:
                timeline = json.load(fp)

        if count is None:
            count = len(timeline)

        remaining_get, reset_time = rate_limit('statuses', 'lookup')

        # Since twitter only allows a certain amount of tweets per request, we cycle through
        for i in range(0, count, self.MAX_TWEETS_PER_REQUEST):
            end = i + self.MAX_TWEETS_PER_REQUEST 
            if end > count:
                end = count
            ids = [tweet['id_str'] for tweet in timeline[i:end]]

            if remaining_get == 0:
                time.sleep(reset_time - time.time() + 1)
                remaining_get, reset_time = rate_limit('statuses', 'lookup')

            tweets = self.twitter.lookup_status(id=','.join(ids), trim_user=True)
            new_timeline.extend(tweets)

        new_timeline = sorted(new_timeline, reverse=True, key=lambda x: x['id'])

        with open(os.path.join('users', 'data', user + '_updated.json'), 'w') as fp:
            json.dump(new_timeline, fp)

    def analyze_user(self, user):
        profile = {}
        timeline = []

        if os.path.isfile(os.path.join('users', 'profiles', user + '.json')):
            with open(os.path.join('users', 'profiles', user + '.json'), 'r') as fp:
                profile = json.load(fp)
        else:
            profile = self.get_user(user)

        if os.path.isfile(os.path.join('users', 'data', user + '.json')):
            with open(os.path.join('users', 'data', user + '.json'), 'r') as fp:
                timeline = json.load(fp)
        else:
            timeline = self.get_user_timeline(user)

        joined = datetime.strptime(profile['created_at'], '%a %b %d %H:%M:%S %z %Y')

        print('days since creation:', abs((joined - datetime.now(timezone.utc)).days))
        print('# of tweets:', profile['statuses_count'])

        popularities = [tweet['favorite_count'] + tweet['retweet_count'] for tweet in timeline]

        popular_threshold = np.percentile(popularities, 90)
        unpopular_threshold = np.percentile(popularities, 10)

        popular_tweets = [(tweet['text'], tweet['favorite_count'] + tweet['retweet_count'])
                            for tweet in timeline 
                            if tweet['favorite_count'] + tweet['retweet_count'] > popular_threshold]

        popular_tweets.sort(key=lambda x: x[1], reverse=True) 

        unpopular_tweets = [(tweet['text'], tweet['favorite_count'] + tweet['retweet_count'])
                                for tweet in timeline 
                                if tweet['favorite_count'] + tweet['retweet_count'] < unpopular_threshold] 

        unpopular_tweets.sort(key=lambda x: x[1])

        mentions = self.twitter.search(q='@' + user, count=100, result_type='mixed')

        with open(os.path.join('users', 'mentions', user + '.json'), 'w') as fp:
            json.dump(mentions, fp)

        sar = SentimentIntensityAnalyzer()

        pos_mentions = 0
        neu_mentions = 0
        neg_mentions = 0

        for tweet in mentions['statuses']:
            senti = sar.polarity_scores(tweet['text'])

            if senti['compound'] >= 0.1:
                pos_mentions += 1
            elif senti['compound'] <= -0.1:
                neg_mentions += 1
            else:
                neu_mentions += 1

        print('mentions | positive: {}, neutral: {}, negative: {}'.format(pos_mentions, neu_mentions, neg_mentions))
