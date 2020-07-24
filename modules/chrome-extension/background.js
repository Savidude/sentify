console.log("Background running");

chrome.browserAction.onClicked.addListener(buttonClicked);

function buttonClicked(tab) {
    console.log("Button clicked!");
    console.log(tab);

    let msg = {
        text: "hello"
    }
    chrome.tabs.sendMessage(tab.id, msg);
}
