import { signup, getSession } from "./app.js";

window.addEventListener("DOMContentLoaded", async () => {
  const s = await getSession();
  if (s) window.location.href = "./admin.html";

  const form = document.getElementById("signupForm");
  const err = document.getElementById("error");
  const ok = document.getElementById("ok");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    ok.textContent = "";
    const name = form.name.value.trim();
    const email = form.email.value.trim();
    const password = form.password.value;
    const confirm = form.confirm.value;
    if (password.length < 8) { err.textContent = "Password must be at least 8 characters."; return; }
    if (password !== confirm) { err.textContent = "Passwords do not match."; return; }
    form.querySelector("button[type=submit]").disabled = true;
    try {
      await signup({ name, email, password });
      ok.textContent = "Account created! Redirecting…";
      setTimeout(() => location.href = "./home.html", 700);
    } catch (ex) {
      err.textContent = ex.message;
    } finally {
      form.querySelector("button[type=submit]").disabled = false;
    }
  });
});
