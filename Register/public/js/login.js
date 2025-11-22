import { login, getSession } from "./app.js";

window.addEventListener("DOMContentLoaded", async () => {
  const s = await getSession().catch(()=>null);
  if (s && s.role === "admin") {
    window.location.href = "./admin.html";
    return;
  }
  if (s) {
    window.location.href = "./home.html";
    return;
  }

  const form = document.getElementById("loginForm");
  const err = document.getElementById("error");
  const guestBtn = document.getElementById("guest");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    const email = form.email.value.trim();
    const password = form.password.value;
    form.querySelector("button[type=submit]").disabled = true;
    try {
      const user = await login({ email, password });
      if (!user) throw new Error("Login failed.");
      if (user.role === "admin") window.location.href = "./admin.html";
      else window.location.href = "./home.html";
    } catch (ex) {
      err.textContent = ex.message;
    } finally {
      form.querySelector("button[type=submit]").disabled = false;
    }
  });

  guestBtn?.addEventListener("click", () => {
    window.location.href = "./home.html?guest=1";
  });

  document.getElementById("goto-signup")?.addEventListener("click", ()=>{ location.href = "./signup.html"; });
});
