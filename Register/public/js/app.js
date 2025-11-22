/**
 * YOBA auth — API-first + fallback demo
 * CSP-safe: all scripts are external (no inline)
 */
const USE_API = true;

async function api(path, opts={}){
  const res = await fetch(path, {credentials:'include', headers:{'Content-Type':'application/json'}, ...opts});
  if(!res.ok){
    const msg = await res.text().catch(()=>res.statusText);
    throw new Error(msg || `HTTP ${res.status}`);
  }
  const text = await res.text();
  try{ return text ? JSON.parse(text) : null; }catch{ return text; }
}

// Public API used by page scripts
export async function signup({name,email,password}){
  if(USE_API){ return api('/api/auth/signup',{method:'POST', body:JSON.stringify({name,email,password})}); }
  return signupLocal({name,email,password});
}
export async function login({email,password}){
  if(USE_API){ const data = await api('/api/auth/login',{method:'POST', body:JSON.stringify({email,password})}); return data.user; }
  return loginLocal({email,password});
}
export async function getSession(){
  if(USE_API){ try{ return await api('/api/auth/me'); }catch{ return null; } }
  return getSessionLocal();
}
export async function logout(){
  if(USE_API){ await api('/api/auth/logout',{method:'POST'}); return; }
  return logoutLocal();
}
export async function listUsers(){
  if(USE_API){ return api('/api/users'); }
  return listUsersLocal();
}
export async function updateUser(id, patch){
  if(USE_API){ return api(`/api/users/${id}`, {method:'PATCH', body:JSON.stringify(patch)}); }
  return updateUserLocal(id, patch);
}
export async function deleteUser(id){
  if(USE_API){ return api(`/api/users/${id}`, {method:'DELETE'}); }
  return deleteUserLocal(id);
}

// ======= Local demo fallback (for offline dev) =======
const STORAGE_KEY = 'yoba_users';
const SESSION_KEY = 'yoba_session';

(async function seedAdmin(){
  try{ await api('/api/health'); return; }catch{/* no backend */}
  const users = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  if(!users.find(u => u.email === 'admin@yoba.fit')){
    users.push({ id: crypto.randomUUID(), name: 'YOBA Admin', email: 'admin@yoba.fit', passwordHash: 'demo:admin123', role: 'admin', createdAt: new Date().toISOString() });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(users));
  }
})();

async function hashPassword(pw){
  const enc = new TextEncoder().encode(pw);
  const buf = await crypto.subtle.digest('SHA-256', enc);
  return Array.from(new Uint8Array(buf)).map(b=>b.toString(16).padStart(2,'0')).join('');
}
async function signupLocal({name,email,password}){
  const users = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  if(users.find(u=>u.email.toLowerCase()===email.toLowerCase())) throw new Error('Email is already registered.');
  const user = { id: crypto.randomUUID(), name, email, passwordHash: await hashPassword(password), role:'user', createdAt:new Date().toISOString() };
  users.push(user); localStorage.setItem(STORAGE_KEY, JSON.stringify(users)); localStorage.setItem(SESSION_KEY, JSON.stringify({ userId: user.id }));
  return user;
}
async function loginLocal({email,password}){
  const users = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  const user = users.find(u=>u.email.toLowerCase()===email.toLowerCase());
  if(!user) throw new Error('Invalid email or password.');
  const ok = user.passwordHash === 'demo:admin123' ? (password==='admin123') : (await hashPassword(password)===user.passwordHash);
  if(!ok) throw new Error('Invalid email or password.');
  localStorage.setItem(SESSION_KEY, JSON.stringify({ userId: user.id }));
  return user;
}
function getSessionLocal(){
  const s = localStorage.getItem(SESSION_KEY); if(!s) return null; const {userId}=JSON.parse(s);
  const users = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); return users.find(u=>u.id===userId)||null;
}
function logoutLocal(){ localStorage.removeItem(SESSION_KEY); }
function listUsersLocal(){ return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
function updateUserLocal(id, patch){ const users=listUsersLocal(); const i=users.findIndex(u=>u.id===id); if(i===-1) throw new Error('User not found'); users[i]={...users[i],...patch}; localStorage.setItem(STORAGE_KEY, JSON.stringify(users)); return users[i]; }
function deleteUserLocal(id){ const me=getSessionLocal(); if(me&&me.id===id) throw new Error("You can't delete your own account."); const users=listUsersLocal().filter(u=>u.id!==id); localStorage.setItem(STORAGE_KEY, JSON.stringify(users)); }
