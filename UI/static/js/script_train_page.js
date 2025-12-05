// UI/static/js/script_train_page.js
//=== Variables & State ===//
let recognition;
let bodyScanListening = false;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let detectedText = document.getElementById("detected-text");
let cornerPic = document.getElementById("cornerPic");
let playerMuteIcon = document.getElementById("mutePlayer");
let trainerMuteIcon = document.getElementById("muteTrainer");

let selectedPoses = JSON.parse(sessionStorage.getItem("selectedPoses") || "[]");
let currentIndex = 0;
let poseStatus = []; // "pending", "done", "failed"
let poseAttempts = []; // number of tries per pose

let maxAttempts = 3;
let numOfSeconds = 5;
let perfectCount = 0;
let instructionConfig = "numerical";
let emphasisesConfig = "all";

const posesListEl = document.getElementById("posesList");
const currentPoseGif = document.getElementById("currentPoseGif");
let counter = 0;

//=== Camera Setup ===//
async function initCamera() {
  try {
    let stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
      width: { min: 1280, ideal: 1280, max: 1280 },
      height: { min: 720, ideal: 720, max: 720 },
      facingMode: "user",
    });
    video.srcObject = stream;
    video.muted = true;
    video.play();
  } catch (e) {
    detectedText.className = "red";
    detectedText.textContent = "Camera error: " + e.message + "...";
  }
}

async function loadConfig() {
  try {
    const res = await fetch("/get_config"); // request from server
    if (!res.ok) throw new Error("Config not found");
    const data = await res.json();

    // Parse integers
    const time = parseInt(data.time, 10);
    let attempts;
    if (data.attempts === "infinity") {
      attempts = Infinity;
    } else {
      attempts = parseInt(data.attempts, 10);
    }

    // Strings
    const instructions = data.instructions;
    const emphasises = data.emphasises;

    console.log({ time, attempts, instructions, emphasises });
    return { time, attempts, instructions, emphasises };
  } catch (err) {
    console.error("Error loading config:", err);
    return null;
  }
}

loadConfig().then((config) => {
  if (config) {
    maxAttempts = config.attempts;
    numOfSeconds = config.time;
    instructionConfig = config.instructions;
    emphasisesConfig = config.emphasises;
  }
});

initCamera();

//=== Instructions & Body Scan ===//
function showInstructions() {
  cornerPic.src = "/static/art/instr.png";
  currentPoseGif.src = "/static/art/standin.gif";
  counter++;
  detectedText.className = "white";
  detectedText.textContent =
    counter === 1
      ? "Hello! We need to scan your whole body before the training starts. Are you ready?"
      : "Ready?";

  let utterance = new SpeechSynthesisUtterance(detectedText.textContent);
  utterance.lang = "en-US";
  speechSynthesis.speak(utterance);
  utterance.onend = () => startListeningBodyScan();
}

function speak(text) {
  let utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  speechSynthesis.speak(utterance);
}

function startListeningBodyScan() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.warn("SpeechRecognition not supported in this browser");
    return;
  }

  if (recognition) {
    try {
      recognition.onresult = null;
      recognition.onend = null;
      recognition.abort();
    } catch (e) {}
  }

  recognition = new SpeechRec();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  bodyScanListening = true;

  detectedText.textContent = "~ listening ~";
  detectedText.className = "yellow";
  playerMuteIcon.src = "/static/art/unmuted.png";
  cornerPic.src = "/static/art/hearing.png";

  recognition.onresult = (event) => {
    const phrase = event.results[0][0].transcript.trim().toLowerCase();
    detectedText.textContent = phrase;

    const saidYes = [
      "yes",
      "yeah",
      "yep",
      "sure",
      "ok",
      "okay",
      "of course",
      "i am",
    ].some((w) => phrase.includes(w));

    if (saidYes) {
      bodyScanListening = false;
      stopListening();
      captureBodyScan();
    } else {
      detectedText.className = "white";
      detectedText.textContent = "Please say 'yes' when you're ready";
      speak(detectedText.textContent);
    }
  };

  recognition.onerror = (e) => {
    console.warn("Speech recognition error:", e);
  };

  recognition.onend = () => {
    if (bodyScanListening) {
      try {
        recognition.start();
      } catch (e) {
        console.warn("Failed to restart recognition:", e);
      }
    }
  };

  recognition.start();
}

function captureBodyScan() {
  canvas.width = 640;
  canvas.height = 480;

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  console.log(
    "Actual webcam resolution:",
    video.videoWidth,
    "x",
    video.videoHeight
  );

  const dataUrl = canvas.toDataURL("image/png");

  fetch("/first_scan", {
    method: "POST",
    body: JSON.stringify({ image: dataUrl }),
    headers: { "Content-Type": "application/json" },
  })
    .then((res) => res.json())
    .then((data) => {
      stopListening();

      if (data.status === "ok") {
        detectedText.className = "white";
        detectedText.textContent = "Body scan successful!";
        cornerPic.src = "/static/art/ok.png";
        document.getElementById("mutePlayer").src = "/static/art/muted.png";
        setupPoseList();
        startNextPose();
      } else if (data.reason === "not_full_body") {
        detectedText.className = "red";
        detectedText.textContent =
          "I can't see your whole body. Please step back so your entire body is inside the frame.";
        cornerPic.src = "/static/art/warning.png";
        speak(detectedText.textContent);

        setTimeout(() => {
          captureBodyScan();
        }, 4000);
      } else {
        detectedText.className = "red";
        detectedText.textContent =
          "Body scan failed. Please say yes to try again.";
        speak(detectedText.textContent);
        cornerPic.src = "/static/art/warning.png";
        setTimeout(() => {
          showInstructions();
        }, 4000);
      }
    })
    .catch((err) => {
      console.error("Body scan error:", err);
      detectedText.className = "red";
      detectedText.textContent = "Upload error";
      cornerPic.src = "/static/art/warning.png";
    });
}

//=== Setup Poses List ===//
function setupPoseList() {
  poseStatus = selectedPoses.map(() => "pending");
  poseAttempts = selectedPoses.map(() => 0);
  posesListEl.innerHTML = "";

  selectedPoses.forEach((p, idx) => {
    const li = document.createElement("li");
    li.className = "pose-list-item";
    li.dataset.index = idx;
    li.style.display = "flex";
    li.style.justifyContent = "space-between";
    li.style.marginBottom = "6px";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = p;

    const statusSpan = document.createElement("span");
    statusSpan.className = "pose-status";
    statusSpan.textContent = "⏳";

    li.appendChild(nameSpan);
    li.appendChild(statusSpan);
    posesListEl.appendChild(li);
  });
}

//=== Pose Flow ===//
function startNextPose() {
  const nextIndex = poseStatus.indexOf("pending");
  if (nextIndex === -1) {
    // Training complete
    detectedText.textContent = "Training complete!";
    speak("Training complete. Well done!");
    cornerPic.src = "/static/art/ok.png";

    // Redirect to score
    setTimeout(() => {
      const totalCount = poseStatus.length * 8;
      sessionStorage.setItem("perfect", perfectCount);
      sessionStorage.setItem("total", totalCount);
      window.location.href = `/score?perfect=${perfectCount}&total=${totalCount}`;
    }, 1500);
    return;
  }

  currentIndex = nextIndex;
  updateCurrentPoseUI();
  startCountdownCapture();
}

function updateCurrentPoseUI() {
  const pose = selectedPoses[currentIndex];
  const safeName = pose.toLowerCase().replace(/\s+/g, "_");
  currentPoseGif.src = `/static/art/${safeName}.gif`;

  Array.from(posesListEl.children).forEach((li, idx) => {
    li.style.background =
      idx === currentIndex ? "rgba(0,0,0,0.08)" : "transparent";
    const status = poseStatus[idx];
    const statusEl = li.querySelector(".pose-status");
    if (status === "done") statusEl.textContent = "✓";
    else if (status === "failed") statusEl.textContent = "✗";
    else statusEl.textContent = "⏳";
  });

  if (poseStatus[currentIndex] === "pending") {
    detectedText.className = "white";
    detectedText.textContent = `Next: ${pose}`;
    cornerPic.src = "/static/art/instr.png";
    speak(`Next exercise: ${pose}`);
  }
}

//=== Countdown & Capture ===//
function startCountdownCapture() {
  const interval = setInterval(() => {
    detectedText.textContent = `Capture in: ${numOfSeconds}`;
    detectedText.className = "gray";
    cornerPic.src = "/static/art/clock.png";
    numOfSeconds--;
    if (numOfSeconds < 0) {
      clearInterval(interval);
      capturePose();
    }
  }, 1000);
}

function updatePerfectCount(currWrongs) {
  let totalJoints = 8;
  perfectCount += totalJoints - currWrongs;
}

async function capturePose() {
  const numFrames = 9;
  const duration = 3000; // 3 seconds total
  const interval = duration / numFrames;
  const images = [];

  canvas.width = 640;
  canvas.height = 480;

  // Capture 9 frames evenly spaced across 3 seconds
  for (let i = 0; i < numFrames; i++) {
    detectedText.textContent =
      "Capturing " + (i + 1) + "/" + numFrames + " photos...";
    cornerPic.src = "/static/art/instr.png";
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const dataUrl = canvas.toDataURL("image/png");
    images.push(dataUrl);

    await new Promise((res) => setTimeout(res, interval));
  }

  detectedText.className = "gray";
  detectedText.textContent = "Analyzing your pose...";
  cornerPic.src = "/static/art/instr.png";

  // Send all images together
  fetch("/upload_burst", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      images,
      pose: selectedPoses[currentIndex].toLowerCase().replace(/\s+/g, "_"),
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "ok") {
        if (data.len > 0) {
          poseAttempts[currentIndex]++;
          detectedText.textContent = `${data.msg}\n(${poseAttempts[currentIndex]}/${maxAttempts}) : ${data.len} problems`;
          cornerPic.src = "/static/art/instr.png";
          speak(data.msg);

          if (poseAttempts[currentIndex] >= maxAttempts) {
            updatePerfectCount(data.len);
            poseStatus[currentIndex] = "failed";
            updateCurrentPoseUI();
            setTimeout(startNextPose, 2067);
          } else {
            setTimeout(startCountdownCapture, 2067);
          }
        }
        else {
          updatePerfectCount(data.len);
          poseStatus[currentIndex] = "done";
          updateCurrentPoseUI();
          detectedText.textContent = "Pose correct!";
          cornerPic.src = "/static/art/ok.png";
          speak("Good job!");
          setTimeout(startNextPose, 2067);
        }
      }
      else {
        detectedText.className = "red";
        detectedText.textContent = data.message || "unknown";
        cornerPic.src = "/static/art/no.png";
      }
    })
    .catch((err) => {
      console.error("Pose upload error:", err);
      detectedText.className = "red";
      detectedText.textContent = "Upload error";
      cornerPic.src = "/static/art/warning.png";
    });
}

//=== Utilities ===//
function stopListening() {
  bodyScanListening = false;
  playerMuteIcon.src = "/static/art/muted.png";
  trainerMuteIcon.src = "/static/art/muted.png";
  if (recognition) {
    try {
      recognition.stop();
    } catch (e) {}
  }
}

document.getElementById("startBtn").onclick = () => {
  const bodyScanDone = poseStatus.length > 0;
  trainerMuteIcon.src = "/static/art/unmuted.png";
  if (bodyScanDone) {
    detectedText.className = "gray";
    detectedText.textContent = "Resuming current pose...";
    cornerPic.src = "/static/art/instr.png";
    updateCurrentPoseUI();
    startCountdownCapture();
  } else {
    showInstructions();
  }
};

document.getElementById("endBtn").onclick = () => {
  speak("Bye bye!");
  trainerMuteIcon.src = "/static/art/unmuted.png";
  stopListening();
  window.location.href = `/welcome`;
};

detectedText.textContent = "Click Start to begin";
