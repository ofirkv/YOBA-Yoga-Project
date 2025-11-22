import { getSession, logout } from "./app.js";

window.addEventListener("DOMContentLoaded", async () => {
  let me = await getSession();
  const isGuest = !me;
  const nameEl = document.getElementById("hello");
  const logoutBtn = document.getElementById("logout");
  if (isGuest) {
    nameEl.textContent = "Guest";
    logoutBtn.style.display = "none";
    const badge = document.getElementById("status");
    if (badge) badge.textContent = "Guest mode — limited access";
  } else {
    nameEl.textContent = me.name || me.email;
    logoutBtn.addEventListener("click", async () => {
      await logout();
      location.href = "./index.html";
    });
  }
});

const readyBtn = document.getElementById("ready");

readyBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:5000";
});