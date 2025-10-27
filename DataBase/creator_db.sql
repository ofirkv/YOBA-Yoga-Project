-- Create database (change name if you want)
CREATE DATABASE yoba_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE yoba_db;

-- users table
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- user_profile table
CREATE TABLE user_profile (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  age INT NULL,
  gender ENUM('male','female','other') NULL,
  weight FLOAT NULL,
  height FLOAT NULL,
  experience_level ENUM('beginner','intermediate','advanced') NULL,
  preferred_length VARCHAR(50) NULL, -- e.g. "2-5 exercises" or numeric
  injuries TEXT NULL, -- comma separated or JSON text
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
