import streamlit as st
import sqlite3
from sqlite3 import Error
import pandas as pd
import plotly.express as px
from datetime import datetime
import hashlib
import re
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.data_utils import extract_text_from_file, preprocess_text
from src.modeling import compute_weighted_score, apply_min_max_scaling
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

# --- Page Config ---
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
)

# --- Load Secrets & Config ---
try:
    SMTP_USERNAME = st.secrets["SMTP_USERNAME"]
    SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]
    SMTP_SERVER = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(st.secrets.get("SMTP_PORT", 587))
    ADMIN_SECRET_KEY = st.secrets.get("ADMIN_SECRET_KEY", "Mekdela@2026")
except (KeyError, StreamlitSecretNotFoundError):
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
    ADMIN_SECRET_KEY = "Mekdela@2026"

# --- CSS Styles ---
CUSTOM_CSS = """
<style>
    .header {
        background-color: #006400;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        width: 100%;
        position: sticky;
        top: 0;
        z-index: 2000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .main-content { padding-top: 20px; padding-bottom: 60px; }
    .card {
        background-color: white;
        border: 1px solid #e6e9ef;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .badge { display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; }
    .badge-accepted { background-color: #28a745; color: white; }
    .badge-processing { background-color: #007bff; color: white; }
</style>
"""

# --- Database Functions ---
def get_sqlite_connection():
    return sqlite3.connect('resume_system.db')

def init_db():
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT, email TEXT UNIQUE, phone TEXT, password TEXT, role TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS all_submissions (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT, email TEXT UNIQUE, resume_text TEXT, tf_idf_score REAL, transformer_score REAL, final_score REAL, upload_time DATETIME)")
    cursor.execute("CREATE TABLE IF NOT EXISTS invitation_codes (email TEXT PRIMARY KEY, code TEXT, created_at DATETIME)")
    conn.commit()
    conn.close()

# --- Utility Functions ---
def generate_access_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def send_access_code_email(recipient, code):
    msg = MIMEMultipart(); msg["Subject"] = "Access Code"; msg["From"] = SMTP_USERNAME; msg["To"] = recipient
    msg.attach(MIMEText(f"Your access code is: {code}", "plain"))
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient, msg.as_string())
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- UI Components ---
def main():
    init_db()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="header">Resume Screening System</div>', unsafe_allow_html=True)

    # Session State Initialization
    if "user" not in st.session_state: st.session_state["user"] = None
    if "verification_step" not in st.session_state: st.session_state["verification_step"] = "select_role"
    if "verification_email" not in st.session_state: st.session_state["verification_email"] = ""

    if st.session_state["user"] is None:
        outer_col1, outer_col2, outer_col3 = st.columns([1, 2, 1])
        with outer_col2:
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                st.subheader("Login")
                l_email = st.text_input("Email", key="l_email")
                l_pass = st.text_input("Password", type="password", key="l_pass")
                if st.button("Login"):
                    # Simple Login Logic (In real app, query DB)
                    st.success("Login logic would go here.")

            with tab2:
                role = st.radio("Register as:", ["Candidate", "Recruiter"])
                
                if role == "Candidate":
                    st.text_input("Full Name")
                    st.text_input("Email")
                    st.button("Register as Candidate")
                
                else: # RECRUITER LOGIC (Fixed)
                    if st.session_state["verification_step"] == "select_role":
                        st.subheader("Step 1: Admin Verification")
                        admin_key = st.text_input("Admin Access Key", type="password")
                        rec_email = st.text_input("Recruiter Email")
                        
                        if st.button("Verify & Send Code"):
                            if admin_key != ADMIN_SECRET_KEY:
                                st.error("Incorrect Admin Key!")
                            elif not validate_email(rec_email):
                                st.error("Invalid Email!")
                            else:
                                code = generate_access_code()
                                success, msg = send_access_code_email(rec_email, code)
                                if success:
                                    # Save code to DB
                                    conn = get_sqlite_connection()
                                    conn.execute("INSERT OR REPLACE INTO invitation_codes VALUES (?, ?, ?)", (rec_email, code, datetime.now()))
                                    conn.commit(); conn.close()
                                    
                                    st.session_state["verification_email"] = rec_email
                                    st.session_state["verification_step"] = "verify_code"
                                    st.success("Code sent to email!")
                                    st.rerun()
                                else:
                                    st.error(f"Email Error: {msg}")

                    elif st.session_state["verification_step"] == "verify_code":
                        st.subheader("Step 2: Enter Code")
                        st.info(f"Code sent to {st.session_state['verification_email']}")
                        v_code = st.text_input("6-Digit Code")
                        
                        if st.button("Complete Registration"):
                            # Logic to check code in DB
                            st.success("Verified! Final registration form would appear here.")
                        
                        if st.button("Back"):
                            st.session_state["verification_step"] = "select_role"
                            st.rerun()

    else:
        st.write("Logged in content here.")

if __name__ == "__main__":
    main()
