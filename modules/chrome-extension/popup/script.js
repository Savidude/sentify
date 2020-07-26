let bgpage = chrome.extension.getBackgroundPage();
let removedTweets = bgpage.removedTweetText;
console.log(removedTweets);

window.onload = function() {
    let removedTweetsDiv = document.getElementById('removed-tweet-list');
    removedTweets.forEach(function (tweetText, index) {
        let p = document.createElement('p');
        p.innerHTML = tweetText;
        removedTweetsDiv.appendChild(p);
    });
};
