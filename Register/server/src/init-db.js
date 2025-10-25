import { query } from "./db.js";
import bcrypt from "bcryptjs";
import { v4 as uuidv4 } from "uuid";

async function main() {
  query(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
  `);

  const { rows } = query("SELECT * FROM users WHERE email = ?", [
    "admin@yoba.fit",
  ]);
  if (rows.length === 0) {
    const hash = await bcrypt.hash("admin123", 12);
    query(
      "INSERT INTO users (id,name,email,password_hash,role) VALUES (?,?,?,?,?)",
      [uuidv4(), "YOBA Admin", "admin@yoba.fit", hash, "admin"]
    );
    console.log("Seeded admin: admin@yoba.fit / admin123");
  }
  console.log("SQLite DB ready (yoba.db)");
}

main();
