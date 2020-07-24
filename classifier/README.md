# Classifier

The Classifier is used to build, train, and utilize a model to classify Tweets according to their sentiment.

## Training the Model

The [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140) is used to train the
Sentiment Analyzer module.

1. Download the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140).
2. Copy the dataset into the `classifier/data` directory.
3. Download and Install the Python Dependencies
```shell
$ pip install -r requirements.txt
```
4. Build the model. This may take up to 10 hours.
```shell
$ python build_model.py
```

Once the model has finished building, it will be available in `data/model.h5`

## Running the Classifier
1. Navigate to the Classifier
2. Install Python Dependencies
3. Run the Sentify Classifier web server

```shell script
$ cd classifier
$ pip install -r requirements.txt
$ python services.py
```
