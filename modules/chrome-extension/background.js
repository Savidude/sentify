removedTweetText = [];

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    console.log(request.text);
    removedTweetText.push(request.text);
});
