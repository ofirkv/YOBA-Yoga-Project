// UI/static/js/script_score.js
function getQueryParams() {
  const params = {};
  location.search
    .slice(1)
    .split("&")
    .forEach((pair) => {
      const [key, value] = pair.split("=");
      if (key) params[key] = decodeURIComponent(value);
    });
  return params;
}

function displayScore() {
  const params = getQueryParams();
  let perfect = parseInt(params.perfect) || 0;
  const total = parseInt(params.total) || 0;
  if (perfect < 0) {
    perfect = 0;
  }
  const scoreText = document.getElementById("scoreText");
  const messageText = document.getElementById("messageText");

  scoreText.textContent = `Score: ${perfect} / ${total}`;

  if (perfect === total && total > 0) {
    messageText.textContent = "Excellent! You nailed all poses!";
  } else if (perfect > 0) {
    messageText.textContent = "Good job! Keep practicing to improve.";
  } else {
    messageText.textContent = "Don't worry! Try again to get better.";
  }
}

function goToHome() {
  window.location.href = "/welcome";
}

function trainAgain() {
  window.location.href = "/train";
}

displayScore();
