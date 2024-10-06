document.addEventListener("DOMContentLoaded", function () {
  console.log("popup.js loaded");
  const convertBtn = document.getElementById("convert");
  const resetBtn = document.getElementById("reset");

  convertBtn.addEventListener("click", function () {
    console.log("convert button clicked");
    chrome.tabs.query(
      {
        active: true,
        currentWindow: true,
      },
      function (tabs) {
        console.log({ tabs });
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: "convert" },
          function (response) {
            console.log(response);
          }
        );
      }
    );
  });

  resetBtn.addEventListener("click", function () {
    console.log("reset button clicked");
    chrome.tabs.query(
      {
        active: true,
        currentWindow: true,
      },
      function (tabs) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: "reset" },
          function (response) {
            console.log(response);
          }
        );
      }
    );
  });
});
