# YOBA — Secure Auth Starter (SQLite + Helmet CSP + Guest Mode)

## What’s inside
- SQLite DB (file-based, no install)
- Helmet with default CSP (`script-src 'self'`) — no inline scripts
- External JS modules per page (login/signup/admin/home)
- Auth endpoints (signup, login, me, logout) + admin users CRUD
- Guest Mode: Home accessible without login

npm rebuild better-sqlite3

## Quick start
```bash
cd server
cp .env.example .env   # optional
npm install
npm run init:db
npm run dev
# open http://localhost:3000
```
Admin seed: `admin@yoba.fit / admin123`

## Guest mode
On the login page click **Continue as Guest** → opens Home without auth.
Admin page remains protected (requires admin session).
