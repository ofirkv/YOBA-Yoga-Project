let recognition;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let detectedText = document.getElementById("detected-text");

// Camera always on
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

// Speech recognition setup
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = function(event) {
        let last = event.results.length - 1;
        let text = event.results[last][0].transcript.trim().toLowerCase();
        detectedText.className = "yellow";
        detectedText.textContent = text;

        // Agreement
        if (["yes","yeah","yep","sure","ok","okay"].some(w => text.includes(w))) {
            capturePhoto();
        }

        // Stop words
        if (["stop","no"].some(w => text.includes(w))) {
            stopListening();
            detectedText.className = "gray";
            detectedText.textContent = "*Silenced*";
        }
    };

    recognition.onerror = function(event){
        console.error("Speech error:", event.error);
        detectedText.className = "red";
        detectedText.textContent = "*Error: " + event.error + "*";
    }
}

// Start listening with TTS
function startListening() {
    detectedText.className = "pink";
    detectedText.textContent = "Are you ready?";
    changeGif();
    const msg = new SpeechSynthesisUtterance(detectedText.textContent);
    msg.lang = "en-US";
    window.speechSynthesis.speak(msg);

    msg.onend = () => {
        detectedText.className = "pink";
        detectedText.textContent = "*Listening*";
        recognition.start();
    };
}

function stopListening() {
    if (recognition) recognition.stop();
    detectedText.className = "gray";
    detectedText.textContent = "*Silenced*";
}

function captureFrame() {
    const videoRatio = video.videoWidth / video.videoHeight;
    const canvasRatio = canvas.width / canvas.height;

    let drawWidth, drawHeight, offsetX = 0, offsetY = 0;

    if (canvasRatio > videoRatio) {
        drawHeight = canvas.height;
        drawWidth = videoRatio * drawHeight;
        offsetX = (canvas.width - drawWidth) / 2;
    } else {
        drawWidth = canvas.width;
        drawHeight = drawWidth / videoRatio;
        offsetY = (canvas.height - drawHeight) / 2;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
}

function capturePhoto() {
    // mirror canvas
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    captureFrame();
    ctx.restore();

    stopListening();

    let dataUrl = canvas.toDataURL("image/png");
    let theHeader = document.getElementById("ttl");

    fetch("/upload", {
        method: "POST",
        body: JSON.stringify({ image: dataUrl }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        console.log("Saved:", data);

        detectedText.className = "pink";
        detectedText.textContent = "*Captured!*";

        if (data.status === "ok") {
            detectedText.className = "pink";
            let instr = data.msg;
            let numWrongs = data.len;

            if(numWrongs === 0) { //No problem :)
                detectedText.textContent = "All good, Great job!";
                const msg = new SpeechSynthesisUtterance(detectedText.textContent);
                msg.lang = "en-US";
                window.speechSynthesis.speak(msg);
            }
            else { //Problem found :(
                detectedText.textContent = "(1/" + numWrongs + ") " + instr;
                const msg = new SpeechSynthesisUtterance(instr);
                msg.lang = "en-US";
                msg.onend = () => {
                    // Countdown before next capture
                    countdownCapture();
                };
                window.speechSynthesis.speak(msg);
            }
        }
        else {
            console.error("Pose error:", data.message);
        }

    })
    .catch(err => console.error("Upload error:", err));
}

function countdownCapture() {
    let count = 5;
    const interval = setInterval(() => {
        detectedText.className = "gray";
        if(count === 1)
            detectedText.className = "red";
        detectedText.textContent = "*Capture in: " + count + "*";
        const msg = new SpeechSynthesisUtterance(count.toString());
        msg.lang = "en-US";
        window.speechSynthesis.speak(msg);
        count--;
        if (count < 0) {
            clearInterval(interval);
            capturePhoto(); // Take next picture
        }
    }, 1000);
}

document.getElementById("startBtn").onclick = startListening;
document.getElementById("stopBtn").onclick = stopListening;