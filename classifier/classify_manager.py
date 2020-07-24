from classify import predict
from model.prediction_result import PredictionResult


def predict_many(classify_request):
    tweets = classify_request.tweets
    predictions = []
    for tweet in tweets:
        prediction = predict(tweet.message)
        prediction_result = PredictionResult(tweet.id, prediction)
        predictions.append(prediction_result.get_dict())
    return predictions
