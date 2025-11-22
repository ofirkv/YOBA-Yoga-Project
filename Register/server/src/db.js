import Database from "better-sqlite3";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dbPath = path.join(__dirname, "../../yoba.db");
export const db = new Database(dbPath);

export function query(sql, params = []) {
  const stmt = db.prepare(sql);
  if (sql.trim().toLowerCase().startsWith("select")) {
    return { rows: stmt.all(...params) };
  } else {
    const result = stmt.run(...params);
    return { rowCount: result.changes, lastInsertRowid: result.lastInsertRowid };
  }
}
