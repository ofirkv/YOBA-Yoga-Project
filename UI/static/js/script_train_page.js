//=== Variables & State ===//
let recognition;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let detectedText = document.getElementById("detected-text");

let selectedPoses = JSON.parse(sessionStorage.getItem('selectedPoses') || '[]');
let currentIndex = 0;
let poseStatus = [];       // "pending", "done", "failed"
let poseAttempts = [];     // number of tries per pose
const maxAttempts = 3;
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
                facingMode: "user"
            });
        video.srcObject = stream;
        video.muted = true;
        video.play();
    } catch (e) {
        detectedText.className = "red";
        detectedText.textContent = "*Camera error: " + e.message + "*";
    }
}
initCamera();

//=== Instructions & Body Scan ===//
function showInstructions() {
    currentPoseGif.src = "/static/art/standin.gif";
    counter++;
    detectedText.className = "pink";
    detectedText.textContent = counter === 1 
        ? "Hello! We need to scan your whole body before the training starts. Are you ready?" 
        : "Ready?";

    let utterance = new SpeechSynthesisUtterance(detectedText.textContent);
    utterance.lang = "en-US";
    speechSynthesis.speak(utterance);
    utterance.onend = () => startListeningBodyScan();
}

function startListeningBodyScan() {
    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    detectedText.className = "yellow";
    detectedText.textContent = "...";

    recognition.onresult = (event) => {
        const phrase = event.results[0][0].transcript.trim().toLowerCase();
        detectedText.textContent = phrase;

        if (["yes","yeah","yep","sure","ok","okay","bailey","ofir","yoav","banana","okay"].some(w => phrase.includes(w))) {
            captureBodyScan();
        } else {
            stopListening();
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

    console.log("Actual webcam resolution:", video.videoWidth, "x", video.videoHeight);

    const dataUrl = canvas.toDataURL("image/png");

    fetch("/first_scan", {
        method: "POST",
        body: JSON.stringify({ image: dataUrl }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        stopListening();
        if (data.status === "ok") {
            detectedText.className = "pink";
            detectedText.textContent = "*Body scan successful!*";
            setupPoseList();
            startNextPose();
        } else {
            detectedText.className = "red";
            detectedText.textContent = "*Body scan failed, click Start again*";
        }
    })
    .catch(err => {
        console.error("Body scan error:", err);
        detectedText.className = "red";
        detectedText.textContent = "*Upload error*";
    });
}

//=== Setup Poses List ===//
function setupPoseList() {
    poseStatus = selectedPoses.map(() => 'pending');
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
        detectedText.textContent = "*Training complete!*";
        speechSynthesis.speak(new SpeechSynthesisUtterance("Training complete. Well done!"));

        // Redirect to score
        setTimeout(() => {
            const perfectCount = poseStatus.filter(s => s === "done").length;
            const totalCount = poseStatus.length;
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
        li.style.background = idx === currentIndex ? "rgba(0,0,0,0.08)" : "transparent";
        const status = poseStatus[idx];
        const statusEl = li.querySelector(".pose-status");
        if (status === "done") statusEl.textContent = "✓";
        else if (status === "failed") statusEl.textContent = "✗";
        else statusEl.textContent = "⏳";
    });

    if (poseStatus[currentIndex] === "pending") {
        detectedText.className = "pink";
        detectedText.textContent = `Next: ${pose}`;
        const msg = new SpeechSynthesisUtterance(`Next exercise: ${pose}`);
        msg.lang = "en-US";
        speechSynthesis.speak(msg);
    }
}

//=== Countdown & Capture ===//
function startCountdownCapture() {
    let count = 5;
    const interval = setInterval(() => {
        detectedText.textContent = `Capture in: ${count}`;
        detectedText.className = "gray";
        count--;
        if (count < 0) {
            clearInterval(interval);
            capturePose();
        }
    }, 1000);
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
        detectedText.textContent = "Capturing " + (i+1) + "/" + numFrames + " photos..." ;
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        const dataUrl = canvas.toDataURL("image/png");
        images.push(dataUrl);

        await new Promise(res => setTimeout(res, interval));
    }

    detectedText.className = "gray";
    detectedText.textContent = "Analyzing your pose...";

    // Send all images together
    fetch("/upload_burst", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            images,
            pose: selectedPoses[currentIndex].toLowerCase().replace(/\s+/g, "_")
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "ok") {
            if (data.len > 0) {
                poseAttempts[currentIndex]++;
                detectedText.textContent = `${data.msg}\n(${poseAttempts[currentIndex]}/${maxAttempts})`;
                speechSynthesis.speak(new SpeechSynthesisUtterance(data.msg));

                if (poseAttempts[currentIndex] >= maxAttempts) {
                    poseStatus[currentIndex] = "failed";
                    updateCurrentPoseUI();
                    setTimeout(startNextPose, 1500);
                } else {
                    setTimeout(startCountdownCapture, 1500);
                }
            } else {
                poseStatus[currentIndex] = "done";
                updateCurrentPoseUI();
                detectedText.textContent = "*Pose correct!*";
                speechSynthesis.speak(new SpeechSynthesisUtterance("Good job!"));
                setTimeout(startNextPose, 1000);
            }
        } else {
            detectedText.className = "red";
            detectedText.textContent = "*Server error* " + (data.message || "unknown");
        }
    })
    .catch(err => {
        console.error("Pose upload error:", err);
        detectedText.className = "red";
        detectedText.textContent = "*Upload error*";
    });
}

//=== Utilities ===//
function stopListening() { if (recognition) recognition.stop(); }
document.getElementById("startBtn").onclick = () => {
    const bodyScanDone = poseStatus.length > 0;
    if (bodyScanDone) {
        detectedText.className = "gray";
        detectedText.textContent = "*Resuming current pose...*";
        updateCurrentPoseUI();
        startCountdownCapture();
    } else {
        showInstructions();
    }
};
detectedText.textContent = "*Click Start to begin*";