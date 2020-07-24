# Sentify

Sentify is a Sentiment Analyzer for Twitter that identifies angry and negative tweets, and removes them from your feed
because you don't deserve that kind of negativity in your life.

## Requirements
* Python 3.7+
* Chrome Browser
* 1GB RAM (If you don't have 1GB RAM, what are you doing with your life?)

### Running the Classifier
1. Navigate to the Classifier
2. Install Python Dependencies
3. Run the Sentify Classifier web server

```shell script
$ cd classifier
$ pip install -r requirements.txt
$ python services.py
```

### Running Sentify Modules

All modules communicate with the Classifier webs erver to analyze tweet sentiments. Once the classifier is running, run
any module to communicate with the Classifier and filter tweets based on sentiment.

Sentify currently supports the following modules:
* [Chrome Extension](modules/chrome-extension/README.md)
