let nonNegativeText = [];

$(window).on('load', function (e) {
    const interval = setInterval(function() {
        let tweetList = getTweetList();
        let contentMapping = getTweetContent(tweetList);
        if (contentMapping.length > 0) {
            classifyMany(contentMapping);
        }
    }, 5000);
});

function getTweetList() {
    let timeline = document.querySelector('[aria-label="Timeline: Your Home Timeline"]');
    return timeline.firstElementChild.firstElementChild;
}

function getTweetContent(tweetList) {
    let tweets = tweetList.childNodes;
    let tweetsArray = Array.from(tweets);

    let id = 0;
    let contentMapping = [];
    tweetsArray.forEach(function (tweet, index) {
        let text = getText(tweet);
        if (text != null && !nonNegativeText.includes(text)) {
            let mapping = {id: id, text: text, content: tweet};
            contentMapping.push(mapping);
            id++;
        }        
    });
    return contentMapping;
}

function getText(tweet) {
    let text;
    try {
        let tc1 = tweet.firstElementChild.firstElementChild.firstElementChild.firstElementChild.firstElementChild.firstElementChild;
        let tc2 = tc1.childNodes[1].childNodes[1].childNodes[1];
        let tc3 = tc2.firstElementChild.firstElementChild;
        text = tc3.textContent
    } catch (eror) {
        text = null;
    }
    return text;
}

function classifyMany(contentMapping) {
    let capturedTweets = [];
    contentMapping.forEach(function(tweetData, index) {
        let data = {id: tweetData.id, message: tweetData.text};
        capturedTweets.push(data);
    });
    let classifyRequest = {tweets: capturedTweets};

    $.ajax({
        url: 'http://localhost:5000/classify',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(classifyRequest),
        dataType: 'json',
        success: function(data){
            removeNegativeContent(contentMapping, data);
        },
        error: function(){
            console.log("Error occured while classifying Tweets");
        },
    });
}

function removeNegativeContent(contentMapping, classificationResults) {
    classificationResults.forEach(function (result, index) {
        let tweetContent = $.grep(contentMapping, function (e) {
            return e.id === result.id;
        });
        tweetContent = tweetContent[0];
        if (result.label === "NEGATIVE"){
            let tweetHTML = tweetContent.content;
            if (tweetHTML.childNodes.length > 0) {
                console.log("Removed tweet: " + tweetContent.text);
                tweetHTML.firstElementChild.remove();
                saveRemovedTweet(tweetContent);
            }
        } else {
            nonNegativeText.push(tweetContent.text);
        }
    });
}

function saveRemovedTweet(tweetContent) {
    chrome.runtime.sendMessage(tweetContent);
}
