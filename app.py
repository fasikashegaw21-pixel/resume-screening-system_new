import streamlit as st
import sqlite3
from sqlite3 import Error
import pandas as pd
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

# --- Page Config ---
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
)

# --- Load Secrets & Config ---
ADMIN_SECRET_KEY = st.secrets.get("ADMIN_SECRET_KEY", "Mekdela@2026")
SMTP_USERNAME = st.secrets.get("SMTP_USERNAME", "")
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# --- Database Functions ---
def get_sqlite_connection():
    return sqlite3.connect('resume_system.db')

def init_db():
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS invitation_codes (email TEXT PRIMARY KEY, code TEXT, created_at DATETIME)")
    conn.commit()
    conn.close()

# --- Utility Functions ---
def generate_access_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def send_access_code_email(recipient, code):
    msg = MIMEMultipart()
    msg["Subject"] = "Your Verification Code"
    msg["From"] = SMTP_USERNAME
    msg["To"] = recipient
    msg.attach(MIMEText(f"Your verification code is: {code}", "plain"))
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
    
    # Session State Initialization
    if "verification_step" not in st.session_state:
        st.session_state["verification_step"] = "select_role"
    if "verification_email" not in st.session_state:
        st.session_state["verification_email"] = ""

    st.title("📄 Resume Screening System")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab2:
        role = st.radio("Register as:", ["Candidate", "Recruiter"], key="role_selector")
        
        if role == "Recruiter":
            # --- STEP 1: ADMIN KEY & EMAIL ---
            if st.session_state["verification_step"] == "select_role":
                st.subheader("Step 1: Admin Verification")
                
                # 'key' መጨመሩ መረጃው እንዳይጠፋ ይረዳል
                adm_key = st.text_input("Admin Access Key", type="password", key="admin_key_input")
                email_input = st.text_input("Recruiter Email", key="rec_email_input")
                
                if st.button("Verify & Send Code"):
                    if adm_key != ADMIN_SECRET_KEY:
                        st.error("Incorrect Admin Key!")
                    elif not validate_email(email_input):
                        st.error("Invalid Email Address!")
                    else:
                        code = generate_access_code()
                        success, msg = send_access_code_email(email_input, code)
                        
                        if success:
                            # ዳታቤዝ ውስጥ ኮዱን ማስቀመጥ
                            conn = get_sqlite_connection()
                            conn.execute("INSERT OR REPLACE INTO invitation_codes VALUES (?, ?, ?)", 
                                         (email_input, code, datetime.now()))
                            conn.commit()
                            conn.close()
                            
                            # ወደ ሚቀጥለው ደረጃ መሸጋገር
                            st.session_state["verification_email"] = email_input
                            st.session_state["verification_step"] = "verify_code"
                            st.success(f"Code sent to {email_input}")
                            st.rerun()
                        else:
                            st.error(f"Failed to send email: {msg}")

            # --- STEP 2: ENTER CODE ---
            elif st.session_state["verification_step"] == "verify_code":
                st.subheader("Step 2: Enter Code")
                st.info(f"Verification code sent to: **{st.session_state['verification_email']}**")
                
                v_code = st.text_input("6-Digit Code", key="v_code_input")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Complete Registration"):
                        # ኮዱን ከዳታቤዝ ጋር ማረጋገጥ
                        conn = get_sqlite_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT code FROM invitation_codes WHERE email=?", (st.session_state["verification_email"],))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result and v_code == result[0]:
                            st.success("✅ Verification Successful! Welcome Recruiter.")
                            # እዚህ ጋር የተጠቃሚውን ሙሉ መረጃ መቀበያ ፎርም ማሳየት ትችላለህ
                        else:
                            st.error("Invalid Code. Please check your email again.")
                
                with col2:
                    if st.button("Back"):
                        st.session_state["verification_step"] = "select_role"
                        st.rerun()

        else:
            st.subheader("Candidate Registration")
            st.text_input("Full Name", key="cand_name")
            st.text_input("Email", key="cand_email")
            st.button("Register")

if __name__ == "__main__":
    main()
