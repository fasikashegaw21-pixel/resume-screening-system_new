-- SQLite Database Schema for Resume Screening System

DROP TABLE IF EXISTS all_submissions;
DROP TABLE IF EXISTS invitation_codes;
DROP TABLE IF EXISTS users;

-- Table structure for table all_submissions
CREATE TABLE all_submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  email TEXT NOT NULL,
  resume_text TEXT,
  tf_idf_score REAL,
  transformer_score REAL,
  final_score REAL,
  upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data for all_submissions
INSERT INTO all_submissions (full_name, email, resume_text, tf_idf_score, transformer_score, final_score, upload_time) VALUES
('alem', 'alem@gmail.com', 'teacher', 0, 0.235936, 0.165155, '2026-04-23 22:22:10'),
('asresach', '11asresach21@gmail.com', 'full stack developer', 1, 1, 1, '2026-04-23 22:22:10');

-- Table structure for table invitation_codes
CREATE TABLE invitation_codes (
  email TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data for invitation_codes
INSERT INTO invitation_codes (email, code, created_at) VALUES
('11asresach21@gmail.com', 'FBZ6OY', '2026-04-23 22:25:28'),
('bekelebelayneh76@gmail.com', 'TX6DRD', '2026-04-23 17:37:08'),
('belayneh76@gmail.com', 'DE27DP', '2026-04-23 17:36:42');

-- Table structure for table users
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  phone TEXT,
  password TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('candidate', 'recruiter'))
);

-- Insert sample data for users
INSERT INTO users (full_name, email, phone, password, role) VALUES
('Belayneh Bekele', 'bekelebelayneh76@gmail.com', '0955072048', '7e922c566c6d1cf6bc96f187260a8e4cf50ba40d1a3977b073e5fa490a49b543', 'recruiter'),
('abebe kebede', 'abebe@gmail.com', '0955667767', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 'candidate'),
('abebe kebede', 'abebekebede@gmail.com', '0955667767', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 'candidate'),
('alem', 'alem@gmail.com', '0923344556', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 'candidate'),
('alemu mekete', 'alemukass@gmail.com', '0955072042', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 'candidate'),
('asresach', '11asresach21@gmail.com', '0955072041', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 'candidate');
