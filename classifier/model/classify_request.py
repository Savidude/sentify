from model.tweet import Tweet


class ClassifyRequest:
    def __init__(self, tweets):
        tweet_list = []
        for tweet in tweets:
            tweet_id = tweet['id']
            message = tweet['message']
            tweet_data = Tweet(tweet_id, message)
            tweet_list.append(tweet_data)
        self.tweets = tweet_list
