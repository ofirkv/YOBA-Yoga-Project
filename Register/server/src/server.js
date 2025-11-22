import 'dotenv/config';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import cookieParser from 'cookie-parser';
import { query } from './db.js';
import { v4 as uuidv4 } from 'uuid';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import path from 'path';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { fileURLToPath } from 'url';

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Security: default Helmet with CSP (no inline allowed)
app.use(helmet());

app.use(morgan('dev'));
app.use(express.json({ limit: '15mb' }));
app.use(cookieParser());
app.use(cors({ origin: [/^http:\/\/localhost:\d+$/], credentials: true }));

// Static frontend
app.use(express.static(path.join(__dirname, '../../public')));

app.get('/api/health', (req,res)=>res.json({ok:true}));

// JWT helpers
function signToken(user){
  return jwt.sign({ sub: user.id, role: user.role, email: user.email }, process.env.JWT_SECRET || 'devsecret', { expiresIn: '7d' });
}
function authRequired(roles){
  return (req,res,next)=>{
    try{
      const token = req.cookies?.yoba_session || (req.headers.authorization||'').replace('Bearer ','');
      if(!token) return res.status(401).send('Unauthorized');
      const payload = jwt.verify(token, process.env.JWT_SECRET || 'devsecret');
      req.user = payload;
      if(roles && roles.length && !roles.includes(payload.role)) return res.status(403).send('Forbidden');
      next();
    }catch{ return res.status(401).send('Unauthorized'); }
  };
}

// Auth routes
app.post('/api/auth/signup', async (req,res)=>{
  const {name,email,password} = req.body||{};
  if(!name||!email||!password) return res.status(400).send('Missing fields');
  const exists = query("SELECT id FROM users WHERE email = ?", [email.toLowerCase()]).rows;
  if(exists.length) return res.status(409).send('Email is already registered.');
  const id = uuidv4();
  const hash = await bcrypt.hash(password, 12);
  query("INSERT INTO users (id,name,email,password_hash,role) VALUES (?,?,?,?,?)", [id,name,email.toLowerCase(),hash,'user']);
  const token = signToken({ id, role:'user', email });
  res.cookie('yoba_session', token, { httpOnly:true, sameSite:'lax', secure:false, maxAge:7*24*3600*1000 });
  res.status(201).json({ id, name, email, role:'user' });
});

app.post('/api/auth/login', async (req,res)=>{
  const {email,password} = req.body||{};
  if(!email||!password) return res.status(400).send('Missing fields');
  const rows = query("SELECT id,name,email,password_hash,role,created_at FROM users WHERE email = ?", [email.toLowerCase()]).rows;
  const user = rows[0];
  if(!user) return res.status(401).send('Invalid email or password.');
  const ok = await bcrypt.compare(password, user.password_hash);
  if(!ok) return res.status(401).send('Invalid email or password.');
  const token = signToken({ id:user.id, role:user.role, email:user.email });
  res.cookie('yoba_session', token, { httpOnly:true, sameSite:'lax', secure:false, maxAge:7*24*3600*1000 });
  res.json({ user: { id:user.id, name:user.name, email:user.email, role:user.role, createdAt:user.created_at } });
});

app.get('/api/auth/me', authRequired([]), async (req,res)=>{
  const rows = query("SELECT id,name,email,role,created_at FROM users WHERE id = ?", [req.user.sub]).rows;
  const me = rows[0];
  if(!me) return res.status(401).send('Unauthorized');
  res.json({ id:me.id, name:me.name, email:me.email, role:me.role, createdAt:me.created_at });
});

app.post('/api/auth/logout', (req,res)=>{
  res.clearCookie('yoba_session');
  res.status(204).end();
});

// Admin routes
app.get('/api/users', authRequired(['admin']), async (req,res)=>{
  const rows = query("SELECT id,name,email,role,created_at FROM users ORDER BY datetime(created_at) DESC").rows;
  res.json(rows.map(r=>({ id:r.id, name:r.name, email:r.email, role:r.role, createdAt:r.created_at })));
});

app.patch('/api/users/:id', authRequired(['admin']), async (req,res)=>{
  const { id } = req.params; const { role, name } = req.body||{};
  const rows = query("SELECT id FROM users WHERE id = ?", [id]).rows;
  if(!rows.length) return res.status(404).send('User not found');
  if(role) query("UPDATE users SET role = ? WHERE id = ?", [role, id]);
  if(name) query("UPDATE users SET name = ? WHERE id = ?", [name, id]);
  const out = query("SELECT id,name,email,role,created_at FROM users WHERE id = ?", [id]).rows[0];
  res.json({ id:out.id, name:out.name, email:out.email, role:out.role, createdAt:out.created_at });
});

app.delete('/api/users/:id', authRequired(['admin']), async (req,res)=>{
  if(req.params.id === req.user.sub) return res.status(400).send("You can't delete your own account.");
  const result = query("DELETE FROM users WHERE id = ?", [req.params.id]);
  if(!result.rowCount) return res.status(404).send('User not found');
  res.status(204).end();
});

const port = process.env.PORT || 3000;
app.listen(port, ()=> console.log(`YOBA server listening on http://localhost:${port}`));

app.get('/health', (req,res)=>res.json({ok:true}));
