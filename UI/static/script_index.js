let recognition;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let detectedText = document.getElementById("detected-text");
let counter = 0

function showInstructions() {
  counter += 1
  detectedText.className = "pink";
    if(counter===1)
        detectedText.textContent = "Hello welcome to Yoba! In order to use the system we will take pictures of you. We need to scan your whole body in the pictures. Are you ready?";
    else
        detectedText.textContent = "Are you ready?";

  const utterance = new SpeechSynthesisUtterance(detectedText.textContent);
  speechSynthesis.speak(utterance);

  utterance.onend = () => {
    startListening();
  };
}

async function initCamera() {
    try {
        let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.muted = true;
        video.play();
    } catch (e) {
        detectedText.className = "red";
        detectedText.textContent = "*Camera error: " + e.message + "*";
    }
}
initCamera();

function startListening() {
  recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  detectedText.className = "pink";
  detectedText.textContent = "*Listening*";

  recognition.onresult = (event) => {
    const phrase = event.results[0][0].transcript.trim().replace(/[.,!?]/g,"").toLowerCase();
    detectedText.className = "yellow";
    detectedText.textContent = phrase;

    if (["yes","yeah","yep","sure","ok","okay"].some(w => phrase.includes(w))) {
        capturePhotoAndCheck();
    } else {
        stopListening();
        detectedText.className = "gray";
        detectedText.textContent = "*Silenced*";
    }
  };

  recognition.onerror = (event) => {
    console.error("Speech error:", event.error);
    detectedText.className = "red";
    detectedText.textContent = "*Error " + event.error + "*";
  };

  recognition.start();
}

function stopListening() {
    if (recognition) recognition.stop();
    detectedText.className = "gray";
    detectedText.textContent = "*Silenced*";
}

function capturePhotoAndCheck() {
    ctx.save();
    ctx.translate(canvas.width,0);
    ctx.scale(-1,1);
    ctx.drawImage(video,0,0,canvas.width,canvas.height);
    ctx.restore();

    let dataUrl = canvas.toDataURL("image/png");

    fetch("/first_scan", {
        method:"POST",
        body: JSON.stringify({ image: dataUrl }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        if(data.status === "ok"){
            stopListening();
            detectedText.className = "pink";
            detectedText.textContent = "*Captured!*";
            window.location.href = "/ready";
        } else {
            stopListening();
            detectedText.className = "pink";
            if(counter===1)
                detectedText.textContent = "Oops! I couldn't scan your whole body. Please click again and find another position.";
            else
                detectedText.textContent = "Try again";
            let msg = new SpeechSynthesisUtterance(detectedText.textContent);
            msg.lang = "en-US";
            window.speechSynthesis.speak(msg);
        }
    })
    .catch(err => console.error("Upload error:", err));
}

document.getElementById("startBtn").onclick = showInstructions;
