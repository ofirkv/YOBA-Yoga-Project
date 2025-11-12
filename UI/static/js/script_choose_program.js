// script_choose_program.js

document.addEventListener('DOMContentLoaded', () => {
  const posesGrid = document.getElementById('posesGrid');
  const selectedList = document.getElementById('selectedList');
  const startBtn = document.getElementById('startWorkout');
  const maxPoses = 5;

  let selectedPoses = [];

  function updateStartButton() {
    startBtn.disabled = selectedPoses.length === 0;
    startBtn.style.opacity = selectedPoses.length === 0 ? 0.5 : 1;
  }

  function addPose(poseId) {
    if (selectedPoses.length >= maxPoses) {
      alert(`You can select up to ${maxPoses} poses only.`);
      return;
    }

    selectedPoses.push(poseId);

    const item = document.createElement('div');
    item.classList.add('selected-item');

    const nameSpan = document.createElement('span');
    nameSpan.textContent = poseId;

    const removeBtn = document.createElement('button');
    removeBtn.classList.add('remove');
    removeBtn.textContent = 'X';
    removeBtn.addEventListener('click', () => {
      const index = selectedPoses.indexOf(poseId);
      if (index > -1) selectedPoses.splice(index, 1);
      item.remove();
      updateStartButton();
    });

    item.appendChild(nameSpan);
    item.appendChild(removeBtn);
    selectedList.appendChild(item);

    updateStartButton();
  }

  posesGrid.addEventListener('click', (e) => {
    const card = e.target.closest('.pose-card');
    if (!card) return;
    const poseId = card.dataset.poseId;
    addPose(poseId);
  });

  // START button handler: save list to sessionStorage and go to body scan page
  startBtn.addEventListener('click', () => {
    if (selectedPoses.length === 0) return;
    // Save selected poses to sessionStorage
    sessionStorage.setItem('selectedPoses', JSON.stringify(selectedPoses));
    // Redirect to scanning page - change path if your server uses a different route
    window.location.href = '/train'; // or 'train_page.html' depending on your routing
  });

  updateStartButton();
});
