document.addEventListener("DOMContentLoaded", function () {
  const colorsDiv = document.getElementById("colors");

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.tabs.sendMessage(
      tabs[0].id,
      { action: "getColors" },
      function (response) {
        const colors = response.colors;
        colors.forEach((color) => {
          const colorDiv = document.createElement("div");
          colorDiv.style.backgroundColor = rgbToHex(color);
          colorDiv.innerText = rgbToHex(color);
          colorDiv.className = "colorCard";
          colorDiv.onclick = function () {
            navigator.clipboard.writeText(rgbToHex(color));
          };
          colorsDiv.appendChild(colorDiv);
        });
      }
    );
  });
});

function rgbToHex(orig) {
  var a,
    isPercent,
    rgb = orig
      .replace(/\s/g, "")
      .match(/^rgba?\((\d+),(\d+),(\d+),?([^,\s)]+)?/i),
    alpha = ((rgb && rgb[4]) || "").trim(),
    hex = rgb
      ? "#" +
        (rgb[1] | (1 << 8)).toString(16).slice(1) +
        (rgb[2] | (1 << 8)).toString(16).slice(1) +
        (rgb[3] | (1 << 8)).toString(16).slice(1)
      : orig;
  if (alpha !== "") {
    isPercent = alpha.indexOf("%") > -1;
    a = parseFloat(alpha);
    if (!isPercent && a >= 0 && a <= 1) {
      a = Math.round(255 * a);
    } else if (isPercent && a >= 0 && a <= 100) {
      a = Math.round((255 * a) / 100);
    } else {
      a = "";
    }
  }
  if (a) {
    hex += (a | (1 << 8)).toString(16).slice(1);
  }
  return hex.toUpperCase();
}
