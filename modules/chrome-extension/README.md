# Sentify Chrome Extension

The Sentify Chrome Extension filters tweets with a negative sentiment by communicating with the classifier through HTTP.

The Chrome extension reads the tech of teach tweet and passes them over to the classifier to predict if the tweet has a 
begative sentiment. If a tweet is found to have a negative sentiment, it gets removed.

## Running the Sentify Chrome Extension in developer mode

1. Open a Chrome browser.
2. Open the Extension Management page by navigating to `chrome://extensions`.
3. Enable Developer Mode by clicking the toggle switch next to **Developer Mode**.
4. Click the **Load Unpacked** and select the `sentify/modules/chrome-extension` directory.
