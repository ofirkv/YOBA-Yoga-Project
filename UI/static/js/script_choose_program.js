// UI/static/js/script_choose_program.js
document.addEventListener("DOMContentLoaded", () => {
  const posesGrid = document.getElementById("posesGrid");
  const selectedList = document.getElementById("selectedList");
  const startBtn = document.getElementById("startWorkout");
  const filters = document.querySelectorAll(".cat-filter");

  const maxPoses = 5;
  let selectedPoses = [];

  // ---------------------------
  // Enable/disable START button
  // ---------------------------
  function updateStartButton() {
    const disabled = selectedPoses.length === 0;
    startBtn.disabled = disabled;
    startBtn.style.opacity = disabled ? 0.5 : 1;
  }

  // ---------------------------
  // Add pose to list
  // ---------------------------
  function addPose(poseId) {
    if (selectedPoses.length >= maxPoses) {
      alert(`You can select up to ${maxPoses}`);
      return;
    }

    selectedPoses.push(poseId);

    const item = document.createElement("div");
    item.classList.add("selected-item");

    const nameSpan = document.createElement("span");
    nameSpan.textContent = poseId;

    const removeBtn = document.createElement("button");
    removeBtn.classList.add("remove");
    removeBtn.textContent = "X";

    removeBtn.addEventListener("click", () => {
      selectedPoses = selectedPoses.filter((p) => p !== poseId);
      item.remove();
      updateStartButton();
    });

    item.appendChild(nameSpan);
    item.appendChild(removeBtn);
    selectedList.appendChild(item);

    updateStartButton();
  }

  // ---------------------------
  // Pose card click
  // ---------------------------
  posesGrid.addEventListener("click", async (e) => {
    const card = e.target.closest(".pose-card");
    if (!card) return;

    if (!card) return;

    const pose = card.dataset.poseId;
    const category = card.dataset.category;

    const res = await fetch("/check_pose_injuries", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pose, category }),
    });

    const data = await res.json();

    if (data.warning) {
      alert(data.message);
    }

    addPose(pose);
    card.classList.toggle("selected");
  });

  // ---------------------------
  // CATEGORY FILTERING
  // ---------------------------
  function filterPoses() {
    const activeCategories = [...filters]
      .filter((f) => f.checked)
      .map((f) => f.value);

    document.querySelectorAll(".pose-card").forEach((card) => {
      const cat = card.dataset.category;
      card.style.display = activeCategories.includes(cat) ? "block" : "none";
    });
  }

  filters.forEach((filter) => filter.addEventListener("change", filterPoses));

  // initialize
  filterPoses();
  updateStartButton();

  // ---------------------------
  // START BUTTON
  // ---------------------------
  startBtn.addEventListener("click", () => {
    if (selectedPoses.length === 0) return;

    sessionStorage.setItem("selectedPoses", JSON.stringify(selectedPoses));
    window.location.href = "/train";
  });
});

// SAVE CONFIGURATION
const attemptsSlider = document.getElementById("attemptsSlider");
const attemptsValue = document.getElementById("attemptsValue");
const timeSlider = document.getElementById("timeSlider");
const timeValue = document.getElementById("timeValue");

// Update slider labels
attemptsSlider.addEventListener(
  "input",
  () => (attemptsValue.textContent = attemptsSlider.value)
);
timeSlider.addEventListener(
  "input",
  () => (timeValue.textContent = timeSlider.value)
);

// Hide/show attempts slider based on infinity option
document.querySelectorAll('input[name="attemptsOption"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    attemptsSlider.style.display = radio.value === "bar" ? "block" : "none";
    attemptsValue.style.display = radio.value === "bar" ? "inline" : "none";
  });
});

// SAVE CONFIG — send to server
document.getElementById("saveConfigBtn").addEventListener("click", () => {
  const instructions = document.querySelector(
    'input[name="instructions"]:checked'
  ).value;
  const attemptsOption = document.querySelector(
    'input[name="attemptsOption"]:checked'
  ).value;
  const attempts = attemptsOption === "bar" ? attemptsSlider.value : "infinity";
  const time = timeSlider.value;
  const emphasises = document.querySelector(
    'input[name="emphasises"]:checked'
  ).value;

  const configData = { instructions, attempts, time, emphasises };

  fetch("/save_config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(configData),
  })
    .then((res) => res.json())
    .then((data) => alert("Config saved!"))
    .catch((err) => alert("Error saving config"));
});
