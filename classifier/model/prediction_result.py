class PredictionResult:
    def __init__(self, tweet_id, prediction):
        self.id = tweet_id
        self.label = prediction['label']
        self.score = prediction['score']
        self.elapsed_time = prediction['elapsed_time']

    def get_dict(self):
        return {'id': self.id, 'label': self.label, 'score': self.score, 'elapsed_time': self.elapsed_time}
