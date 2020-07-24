import time

import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

KERAS_MODEL_PATH = "data/model.h5"
TOKENIZER_MODEL = "data/tokenizer.pkl"

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# KERAS
SEQUENCE_LENGTH = 300

model = load_model(KERAS_MODEL_PATH)
tokenizer_f = open(TOKENIZER_MODEL, "rb")
tokenizer = pickle.load(tokenizer_f)
tokenizer_f.close()


def decode_sentiment_score(sentiment_score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if sentiment_score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif sentiment_score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if sentiment_score < 0.5 else POSITIVE


def predict(text, include_neutral=True):
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    sentiment_score = model.predict([x_test])[0]
    label = decode_sentiment_score(sentiment_score, include_neutral=include_neutral)

    return {'label': label, 'score': float(sentiment_score), 'elapsed_time': time.time() - start_at}
