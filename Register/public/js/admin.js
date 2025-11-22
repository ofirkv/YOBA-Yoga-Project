import {
  listUsers,
  updateUser,
  deleteUser,
  getSession,
  logout,
} from "./app.js";

function render(users) {
  const tbody = document.querySelector("tbody");
  tbody.innerHTML = users
    .map(
      (u) => `
    <tr>
      <td>${u.name}</td>
      <td>${u.email}</td>
      <td><span class="badge">${u.role}</span></td>
      <td>${new Date(u.createdAt).toLocaleDateString()}</td>
      <td class="util">
        <button class="btn btn-ghost" data-role="${u.id}">Toggle role</button>
        <button class="btn btn-danger" data-del="${u.id}">Delete</button>
      </td>
    </tr>
  `
    )
    .join("");
}

window.addEventListener("DOMContentLoaded", async () => {
  const me = await getSession();
  if (!me || me.role !== "admin") {
    window.location.href = "./index.html";
    return;
  }
  document.getElementById("me").textContent = me.email;
  render(await listUsers());

  const q = document.getElementById("q");
  q.addEventListener("input", async () => {
    const term = q.value.toLowerCase();
    const all = await listUsers();
    render(
      all.filter((u) => `${u.name} ${u.email}`.toLowerCase().includes(term))
    );
  });

  document.body.addEventListener("click", async (e) => {
    const idRole = e.target.getAttribute("data-role");
    const idDel = e.target.getAttribute("data-del");
    if (idRole) {
      const all = await listUsers();
      const u = all.find((x) => x.id === idRole);
      if (u) {
        await updateUser(u.id, { role: u.role === "admin" ? "user" : "admin" });
        render(await listUsers());
      }
    }
    if (idDel) {
      try {
        await deleteUser(idDel);
        render(await listUsers());
      } catch (ex) {
        alert(ex.message);
      }
    }
  });

  document.getElementById("logout").addEventListener("click", async () => {
    await logout();
    location.href = "./index.html";
  });
});
