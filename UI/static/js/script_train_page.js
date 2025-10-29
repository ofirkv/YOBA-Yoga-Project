// script_train_page.js
let recognition;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let detectedText = document.getElementById("detected-text");

let selectedPoses = JSON.parse(sessionStorage.getItem('selectedPoses') || '[]');
let currentIndex = 0;
let poseStatus = []; // "pending" | "done"

const posesListEl = document.getElementById("posesList");
var currentPose = "standin";
const currentPoseGif = document.getElementById("currentPoseGif");
const ttl = document.getElementById("ttl");

// Initialize UI
function renderPosesList() {
    poseStatus = selectedPoses.map(() => 'pending');
    posesListEl.innerHTML = '';

    selectedPoses.forEach((p, idx) => {
        const li = document.createElement('li');
        li.className = 'pose-list-item';
        li.dataset.index = idx;
        li.style.display = "flex";
        li.style.alignItems = "center";
        li.style.gap = "8px";
        li.style.marginBottom = "6px";

        const nameSpan = document.createElement('span');
        nameSpan.textContent = p;
        nameSpan.style.flex = "1";
        nameSpan.style.textAlign = "left";

        const statusSpan = document.createElement('span');
        statusSpan.textContent = '⏳'; // will be replaced with ✓ when done
        statusSpan.className = 'pose-status';
        statusSpan.style.minWidth = "20px";
        statusSpan.style.textAlign = "right";

        li.appendChild(nameSpan);
        li.appendChild(statusSpan);
        posesListEl.appendChild(li);
    });
}

function updatePoseUI() {
    if (selectedPoses.length === 0) {
        currentPose = 'standin'; // no pose
        currentPoseGif.src = "{{ url_for('static', filename='art/standin.gif') }}";
        return;
    }

    const pose = selectedPoses[currentIndex];
    currentPose = (currentIndex + 1) + '. ' + pose;

    // try to set gif by convention: art/<pose_id>.gif (you can adapt names)
    const safeName = pose.toLowerCase().replace(/\s+/g, '_');
    currentPoseGif.src = `/static/art/${safeName}.gif`; // fallback if file missing the browser will ignore

    highlightListItem(currentIndex);
}

function highlightListItem(index) {
    Array.from(posesListEl.children).forEach((li) => {
        if (parseInt(li.dataset.index) === index) {
            li.style.background = 'rgba(0,0,0,0.08)';
        } else {
            li.style.background = 'transparent';
        }

        // update symbol
        const statusSpan = li.querySelector('.pose-status');
        const idx = parseInt(li.dataset.index);
        statusSpan.textContent = poseStatus[idx] === 'done' ? '✓' : '⏳';
    });
}

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
        }
    };

    recognition.onerror = function(event) {
        console.error("Speech error:", event.error);
        detectedText.className = "red";
        detectedText.textContent = "*Error: " + event.error + "*";
    }
}

function startListening() {
    if (selectedPoses.length === 0) {
        detectedText.className = "red";
        detectedText.textContent = "*No poses selected*";
        return;
    }

    // start from first pending pose if not yet started
    if (poseStatus.every(s => s === 'done')) {
        // all done already
        detectedText.className = "pink";
        detectedText.textContent = "*All poses completed*";
        return;
    }

    // find next pending if current is already done
    if (poseStatus.length > 0 && poseStatus[currentIndex] === 'done') {
        const next = poseStatus.indexOf('pending');
        if (next > -1) currentIndex = next;
    }

    updatePoseUI();
    detectedText.className = "pink";
    detectedText.textContent = "Are you ready?";

    // speak then start recognition
    const msg = new SpeechSynthesisUtterance(detectedText.textContent);
    msg.lang = "en-US";
    window.speechSynthesis.speak(msg);
    msg.onend = () => {
        detectedText.className = "pink";
        detectedText.textContent = "*Listening*";
        recognition.start();
    };
}

// startListening();
function stopListening() {
    if (recognition) recognition.stop();
    detectedText.className = "gray";
    detectedText.textContent = "*Silenced*";
}

function captureFrame() {
    // ensure canvas size equals video
    if (!canvas.width) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
    }

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
    fetch("/upload", {
        method: "POST",
        body: JSON.stringify({ image: dataUrl, pose: selectedPoses[currentIndex].toLowerCase().replace(/\s+/g, '_') }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        console.log("Saved:", data);
        detectedText.className = "pink";
        detectedText.textContent = "*Captured!*";

        if (data.status === "ok") {
            let instr = data.msg;
            let numWrongs = data.len;

            if (numWrongs === 0) {
                // successful for this pose
                detectedText.textContent = "All good, Great job!";
                const msg = new SpeechSynthesisUtterance(detectedText.textContent);
                msg.lang = "en-US";
                window.speechSynthesis.speak(msg);

                // mark current pose done
                poseStatus[currentIndex] = 'done';
                highlightListItem(currentIndex);

                // advance to next pending pose or finish
                const nextPending = poseStatus.indexOf('pending');
                if (nextPending === -1) {
                    // finished all poses
                    detectedText.textContent = "*Training complete!*";
                    const doneMsg = new SpeechSynthesisUtterance("Training complete. Well done!");
                    doneMsg.lang = "en-US";
                    window.speechSynthesis.speak(doneMsg);
                    // optionally redirect or show summary
                } else {
                    // move to next pose after short delay and prompt again
                    currentIndex = nextPending;
                    setTimeout(() => { startListening(); }, 1200);
                }

            } else {
                // problem found: server returned instructions about correction
                detectedText.textContent = "(1/" + numWrongs + ") " + instr;
                const msg = new SpeechSynthesisUtterance(instr);
                msg.lang = "en-US";
                msg.onend = () => { countdownCapture(); }; // give user countdown and take next picture automatically
                window.speechSynthesis.speak(msg);
            }

        } else {
            console.error("Pose error:", data.message);
            detectedText.className = "red";
            detectedText.textContent = "*Server error:*" + data.message ;
        }

    })
    .catch(err => {
        console.error("Upload error:", err);
        detectedText.className = "red";
        detectedText.textContent = "*Upload error*";
    });
}

function countdownCapture() {
    let count = 5;
    const interval = setInterval(() => {
        detectedText.className = "gray";
        if (count === 1) detectedText.className = "red";
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

// initial render
renderPosesList();
updatePoseUI();
highlightListItem(currentIndex);

document.getElementById("startBtn").onclick = startListening;
