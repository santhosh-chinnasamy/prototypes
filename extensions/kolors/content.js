chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === "getColors") {
    const elements = document.getElementsByTagName("*");
    const colors = [];
    for (let i = 0; i < elements.length; i++) {
      const element = elements[i];
      const color = window.getComputedStyle(element).getPropertyValue("color");
      if (color && color !== "rgba(0, 0, 0, 0)") {
        colors.push(color);
      }
    }

    const uniqueColors = [...new Set(colors)];
    sendResponse({ colors: uniqueColors });
  }
});
