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

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Header */
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
        left: 0;
        z-index: 2000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        
        
    }
    /* Main content padding to account for sticky header */
    .main-content {
        padding-top: 120px;
        padding-bottom: 60px; /* To prevent content from being hidden behind the footer */
    }
    /* Cards */
    .card {
        background-color: white;
        border: 1px solid #e6e9ef;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
        
    }
    .green-text {
        color: #006400;
        font-weight: bold;
        
    }
    /* Sidebar */
    .sidebar .stRadio > div {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    /* Buttons */
    .submit-btn-container button {
        background-color: #000080; /* Navy Blue */
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .submit-btn-container button:hover {
        background-color: #4169E1; /* Royal Blue */
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        color: blue;
        position: fixed;
        font-weight: bold;
        font-size: 24px;
        bottom: 0;
        width: 100%;
    }
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }
    .badge-accepted {
        background-color: #28a745;
        color: white;
    }
    .badge-processing {
        background-color: #007bff;
        color: white;
        
    }
    
    /* Progress Bar */
    
    .progress-bar {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 20px;
        background-color: #28a745;
        width: 100%; /* Full for accepted */
    }
</style>
"""






# Load SMTP credentials: prioritize st.secrets for deployment, fall back to .env for local testing
try:
    SMTP_USERNAME = st.secrets["SMTP_USERNAME"]
    SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]
    SMTP_SERVER = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(st.secrets.get("SMTP_PORT", 587))
except (KeyError, StreamlitSecretNotFoundError):
    # Fall back to environment variables with .env loading
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))


# --- SQLite Database Functions ---
def get_sqlite_connection():
    """Establish a connection to the SQLite database."""
    try:
        connection = sqlite3.connect('resume_system.db')
        return connection
    except Error as e:
        st.error(f"Error connecting to SQLite: {e}")
        return None

def init_db():
    """Initialize SQLite database and ensure tables exist."""
    connection = get_sqlite_connection()
    if connection:
        cursor = connection.cursor()
        try:
            # Create users table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('candidate', 'recruiter'))
                )
            """)
            # Create all_submissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS all_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    resume_text TEXT,
                    tf_idf_score REAL,
                    transformer_score REAL,
                    final_score REAL,
                    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create invitation_codes table for recruiter pre-verification
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invitation_codes (
                    email TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()
        except Error as e:
            st.error(f"Error creating tables: {e}")
        finally:
            cursor.close()
            connection.close()

def load_submissions() -> pd.DataFrame:
    """Load all submissions from the SQLite database as a Pandas DataFrame."""
    connection = get_sqlite_connection()
    if connection:
        try:
            df = pd.read_sql_query("SELECT full_name AS Name, email AS Email, resume_text AS Resume_Text, tf_idf_score, transformer_score, final_score, upload_time FROM all_submissions", connection)
            return df
        except Error as e:
            st.error(f"Error loading submissions: {e}")
            return pd.DataFrame()
        finally:
            connection.close()
    return pd.DataFrame()

def save_submission(name: str, email: str, resume_text: str, tf_idf_score: float = None, transformer_score: float = None, final_score: float = None):
    """Save a new submission to the SQLite database."""
    connection = get_sqlite_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO all_submissions (full_name, email, resume_text, tf_idf_score, transformer_score, final_score, upload_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, email, resume_text, tf_idf_score, transformer_score, final_score, datetime.now()))
            connection.commit()
        except Error as e:
            st.error(f"Error saving submission: {e}")
        finally:
            cursor.close()
            connection.close()

def find_submission_by_email(email: str) -> bool:
    """Return True if a submission exists in the database for the provided email."""
    connection = get_sqlite_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1 FROM all_submissions WHERE email = ? LIMIT 1", (email,))
            result = cursor.fetchone()
            return result is not None
        except Error as e:
            st.error(f"Error checking submission: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False


# --- Pre-Verification Functions for Recruiters ---
def generate_access_code():
    """Generate a random 6-character alphanumeric code."""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(6))


def save_invitation_code(email: str, code: str):
    """Save or update invitation code in the database."""
    connection = get_sqlite_connection()
    if connection:
        cursor = connection.cursor()
        try:
            # Use INSERT OR REPLACE to handle existing emails
            cursor.execute("""
                INSERT OR REPLACE INTO invitation_codes (email, code, created_at)
                VALUES (?, ?, ?)
            """, (email, code, datetime.now()))
            connection.commit()
            return True
        except Error as e:
            st.error(f"Error saving invitation code: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False


def verify_invitation_code(email: str, code: str) -> bool:
    """Verify if the entered code matches the stored code for the email."""
    connection = get_sqlite_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT code FROM invitation_codes
                WHERE email = ? AND created_at >= datetime('now', '-24 hours')
            """, (email,))
            result = cursor.fetchone()
            if result and result[0] == code:
                return True
            return False
        except Error as e:
            st.error(f"Error verifying code: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False


def send_access_code_email(recipient_email: str, access_code: str):
    """Send the access code to the recruiter's email."""
    message = MIMEMultipart("alternative")
    message["Subject"] = "Resume Screening System - Access Code"
    message["From"] = SMTP_USERNAME
    message["To"] = recipient_email

    html = f"""
    <html>
    <body>
        <h3>Recruiter Access Code</h3>
        <p>You have requested access to register as a recruiter in the Resume Screening System.</p>
        <div style="border-left: 5px solid #007bff; padding: 15px; background: #f8f9fa; margin: 20px 0;">
            <h4 style="margin-top: 0; color: #007bff;">Your Access Code:</h4>
            <p style="font-size: 24px; font-weight: bold; color: #dc3545; letter-spacing: 2px;">{access_code}</p>
            <p style="margin-bottom: 0;"><small>This code is valid for 24 hours.</small></p>
        </div>
        <p>If you did not request this code, please ignore this email.</p>
        <p>Best regards,<br>Resume Screening System Team</p>
    </body></html>
    """
    message.attach(MIMEText(html, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient_email, message.as_string())
        return True, "Access code sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    if not email.strip():
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def register_user(name: str, email: str, phone: str, password: str, role: str):
    """Register a new user."""
    if not validate_email(email):
        return False, "Invalid email address."

    connection = get_sqlite_connection()
    if not connection:
        return False, "Database connection failed."

    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (full_name, email, phone, password, role) VALUES (?, ?, ?, ?, ?)",
            (name, email.lower().strip(), phone, hash_password(password.strip()), role),
        )
        connection.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    except Error as e:
        return False, f"Registration failed: {str(e)}"
    finally:
        cursor.close()
        connection.close()


def login_user(email: str, password: str) -> dict or None:
    """Login user and return user info if successful."""
    connection = get_sqlite_connection()
    if not connection:
        return None

    cursor = connection.cursor()
    try:
        cursor.execute("SELECT full_name, email, phone, role FROM users WHERE email = ? AND password = ?",
                      (email.lower().strip(), hash_password(password.strip())))
        result = cursor.fetchone()
        if result:
            return {"name": result[0], "email": result[1], "phone": result[2], "role": result[3]}
        return None
    except Error as e:
        st.error(f"Login failed: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

@st.cache_resource
def build_models(resume_texts):
    """Build TF-IDF and Transformer models."""
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    vectorizer.fit(resume_texts)
    transformer = SentenceTransformer("all-mpnet-base-v2")
    return vectorizer, transformer

def score_candidates(df, job_description, vectorizer, transformer):
    """Score all candidates against the job description."""
    clean_job = preprocess_text(job_description)
    job_vector = vectorizer.transform([clean_job])
    job_embedding = transformer.encode([clean_job], convert_to_numpy=True)[0]

    results = []
    for idx, row in df.iterrows():
        clean_resume = preprocess_text(row["Resume_Text"])
        tfidf_score = cosine_similarity(vectorizer.transform([clean_resume]), job_vector)[0, 0]
        transformer_score = cosine_similarity(transformer.encode([clean_resume], convert_to_numpy=True), [job_embedding])[0, 0]
        years_of_experience = float(
            row.get("Years of Experience", row.get("Years_of_Experience", row.get("experience_years", 0))) or 0
        )
        final_score = compute_weighted_score(
            tfidf_score,
            transformer_score,
            years_of_experience=years_of_experience,
        )
        results.append({
            "Name": row["Name"],
            "Email": row["Email"],
            "Resume_Text": row["Resume_Text"],
            "upload_time": row.get("upload_time", row.get("Timestamp", None)),
            "Years of Experience": years_of_experience,
            "TF-IDF Score": tfidf_score,
            "Transformer Score": transformer_score,
            "Final Score": final_score,
        })
    results_df = pd.DataFrame(results)
    return results_df.sort_values(["Final Score", "upload_time"], ascending=[False, False]).reset_index(drop=True)


def send_acceptance_email(recipient_email: str, candidate_name: str, position: str, recruiter_name: str, custom_message: str = "") -> tuple[bool, str]:
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return False, "SMTP credentials are not configured."

    message = MIMEMultipart("alternative")
    message["Subject"] = f"Application Update: You are accepted for {position}"
    message["From"] = SMTP_USERNAME
    message["To"] = recipient_email

    text = (
        f"Hello {candidate_name},\n\n"
        f"Congratulations! Based on the job requirements you are among the top matched candidates for the position: {position}.\n\n"
        "A recruiter has reviewed your profile and marked you as accepted.\n\n"
    )
    if custom_message:
        text += f"{custom_message}\n\n"
    text += (
        "Please stay tuned for the next steps.\n\n"
        "Best regards,\n"
        f"{recruiter_name}\n"
        "Resume Screening Team"
    )

    html = (
        f"<html><body><p>Hello {candidate_name},</p>"
        f"<p>Congratulations! Based on the job requirements you are among the top matched candidates for the position: <strong>{position}</strong>.</p>"
        "<p>A recruiter has reviewed your profile and marked you as <strong>accepted</strong>.</p>"
    )
    if custom_message:
        html += f"<p>{custom_message}</p>"
    html += (
        "<p>Please stay tuned for the next steps.</p>"
        "<p>Best regards,<br/>"
        f"{recruiter_name}<br/>Resume Screening Team</p>"
        "</body></html>"
    )

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient_email, message.as_string())
        return True, ""
    except Exception as exc:
        return False, str(exc)


def candidate_dashboard(user):
    st.header(f"Welcome, {user['name']} (Candidate)")
    
    # Display status badge
    accepted_emails = st.session_state.get("accepted_emails", [])
    if user['email'] in accepted_emails:
        st.markdown('<span class="badge badge-accepted">Accepted ✅</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-processing">Under Review ⏳</span>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="green-text"><strong>Candidate Submission</strong></div>', unsafe_allow_html=True)
        st.write("Submit your resume to be reviewed by recruiters.")

        uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])

        st.markdown('<div class="submit-btn-container">', unsafe_allow_html=True)
        if st.button("Submit CV", key="submit_cv"):
            if not uploaded_file:
                st.error("Please upload your CV.")
            else:
                with st.spinner("Processing your CV..."):
                    resume_text = extract_text_from_file(uploaded_file)
                    save_submission(user['name'], user['email'], resume_text)
                st.success("Application Received! Your CV has been submitted successfully.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="green-text"><strong>Check My Status</strong></div>', unsafe_allow_html=True)
        if st.button("Check My Status", key="check_status"):
            if find_submission_by_email(user['email']):
                message = "Status: Received. Your application is in the system."
                if "scores_df" in st.session_state:
                    score_row = st.session_state["scores_df"][st.session_state["scores_df"]["Email"] == user['email']]
                    if not score_row.empty:
                        score_value = score_row.iloc[0]["Final Score"]
                        message = f"Status: Received. Your match score is {score_value:.1%}."
                        st.markdown(f'<span class="badge badge-accepted">Accepted</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge badge-processing">Processing</span>', unsafe_allow_html=True)
                st.info(message)
            else:
                st.warning("No submission found for your email.")
        st.markdown('</div>', unsafe_allow_html=True)


def recruiter_dashboard(user):
    st.header(f"Welcome, {user['name']} (Recruiter)")

    submissions_df = load_submissions()
    if submissions_df.empty:
        st.info("No submissions yet.")
    else:
        st.write(f"Total Submissions: {len(submissions_df)}")

        job_title = st.text_input("Position / Job Title", value="Best matched role")
        job_description = st.text_area("Enter Job Description")
        num_candidates = st.number_input(
            "Number of candidates to accept",
            min_value=1,
            max_value=max(1, len(submissions_df)),
            value=1,
            step=1,
        )

        if st.button("Compute Rankings"):
            if not job_description.strip():
                st.error("Please enter a job description.")
            elif not job_title.strip():
                st.error("Please enter a job title.")
            else:
                resume_texts = [preprocess_text(text) for text in submissions_df["Resume_Text"]]
                vectorizer, transformer = build_models(resume_texts)
                scores_df = score_candidates(submissions_df, job_description, vectorizer, transformer)

                # Save scores to database
                for _, row in scores_df.iterrows():
                    save_submission(
                        name=row["Name"],
                        email=row["Email"],
                        resume_text=row["Resume_Text"],
                        tf_idf_score=row["TF-IDF Score"],
                        transformer_score=row["Transformer Score"],
                        final_score=row["Final Score"]
                    )

                st.session_state["scores_df"] = scores_df
                st.session_state["job_title"] = job_title
                st.session_state["num_candidates"] = int(num_candidates)

        if "scores_df" in st.session_state:
            scores_df = st.session_state["scores_df"]
            job_title = st.session_state.get("job_title", "Best matched role")
            top_n = min(int(st.session_state.get("num_candidates", 1)), len(scores_df))
            selected_df = scores_df.head(top_n)
            st.session_state["accepted_emails"] = selected_df["Email"].tolist()

            tab1, tab2 = st.tabs(["📊 Table View", "📈 Visual Analysis"])

            with tab1:
                st.subheader("Ranked Candidates")
                st.write(f"Top {top_n} candidate(s) for '{job_title}'")
                st.dataframe(style_scores_table(scores_df))

                st.markdown("---")
                st.subheader("Accepted Candidate Emails")
                st.dataframe(
                    selected_df[["Name", "Email", "Final Score"]]
                    .style.format({"Final Score": "{:.2%}"})
                )

                job_position = st.text_input("Position", value=job_title, key="accepted_job_position")
                custom_message = st.text_area("Custom Message to Accepted Candidates", placeholder="e.g., Please attend the interview on [date] at [time].", height=100)

                if st.button("Send acceptance to top candidates"):
                    if not SMTP_USERNAME or not SMTP_PASSWORD:
                        st.error(
                            "SMTP credentials are not configured. For deployment, set them in Streamlit secrets. For local testing, set SMTP_USERNAME and SMTP_PASSWORD environment variables or create a .env file."
                        )
                    else:
                        sent = []
                        failed = []
                        for _, row in selected_df.iterrows():
                            success, error_message = send_acceptance_email(
                                recipient_email=row["Email"],
                                candidate_name=row["Name"],
                                position=job_position,
                                recruiter_name=user["name"],
                                custom_message=custom_message,
                            )
                            if success:
                                sent.append(row["Email"])
                            else:
                                failed.append(f"{row['Email']} ({error_message})")

                        if sent:
                            st.success(f"Acceptance  sent to: {', '.join(sent)}")
                        if failed:
                            st.error(f"Failed to send email to: {', '.join(failed)}")

            with tab2:
                st.subheader("Candidate Comparison")
                compare_df = scores_df.melt(
                    id_vars=["Name"],
                    value_vars=["TF-IDF Score", "Transformer Score"],
                    var_name="Score Type",
                    value_name="Score",
                )
                fig = px.bar(
                    compare_df,
                    x="Name",
                    y="Score",
                    color="Score Type",
                    barmode="group",
                    title="Candidate Score Comparison",
                    text="Score",
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    yaxis_title="Score",
                    legend_title_text="Score Type",
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig)


def get_score_color(score: float) -> str:
    if score >= 0.70:
        return "background-color: #c6efce; color: #006100"
    if score >= 0.40:
        return "background-color: #fff2cc; color: #9c6500"
    return "background-color: #f8cbad; color: #9c0006"


def highlight_row_based_on_final(row: pd.Series) -> list[str]:
    style = get_score_color(row["Final Score"])
    return [style] * len(row)


def style_scores_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        df.style
          .apply(highlight_row_based_on_final, axis=1)
          .format({
              "TF-IDF Score": "{:.2%}",
              "Transformer Score": "{:.2%}",
              "Final Score": "{:.2%}"
          })
          .set_table_styles([
              {
                  "selector": "th",
                  "props": [
                      ("background-color", "#1f2937"),
                      ("color", "white"),
                      ("font-weight", "bold"),
                      ("text-align", "center")
                  ]
              }
          ])
    )


def main():
    # Initialize database
    init_db()
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header"> Resume Screening System</div>', unsafe_allow_html=True)
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Check if user is logged in
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None:
        # Login/Register Page
        st.markdown("<h1 style='text-align: center;'>Welcome to Resume Screening System</h1>", unsafe_allow_html=True)

        outer_col1, outer_col2, outer_col3 = st.columns([1, 2, 1])
        with outer_col2:
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

            with tab1:
                st.subheader("Login")
                if st.session_state.get("registration_message"):
                    st.success(st.session_state["registration_message"])
                email = st.text_input("Email", key="login_email", autocomplete="email")
                password = st.text_input("Password", type="password", key="login_password", autocomplete="current-password")
                if st.button("Login"):
                    user = login_user(email, password)
                    if user:
                        st.session_state["user"] = user
                        st.success("Logged in successfully!")
                        st.session_state.pop("registration_message", None)
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

            with tab2:
                st.subheader("Register")

                # Initialize session state for pre-verification
                if "verification_step" not in st.session_state:
                    st.session_state["verification_step"] = "select_role"
                if "verification_email" not in st.session_state:
                    st.session_state["verification_email"] = ""

                role = st.radio("Register as:", ["👤 Candidate", "🔑 Recruiter"], key="reg_role")

                if role == "👤 Candidate":
                    # Simple registration for candidates
                    st.markdown("### Candidate Registration")
                    name = st.text_input("Full Name", key="reg_name", autocomplete="name")
                    email = st.text_input("Email", key="reg_email", autocomplete="email")
                    phone = st.text_input("Phone", key="reg_phone", autocomplete="tel")
                    password = st.text_input("Password", type="password", key="reg_password", autocomplete="new-password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm", autocomplete="new-password")

                    if st.button("Register as Candidate"):
                        if password != confirm_password:
                            st.error("Passwords do not match.")
                        elif not all([name, email, phone, password]):
                            st.error("Please fill all fields.")
                        elif not validate_email(email):
                            st.error("Please enter a valid email address.")
                        else:
                            success, message = register_user(name, email, phone, password, "candidate")
                            if success:
                                st.session_state["registration_message"] = "Registration successful! Please login."
                                # Reset form
                                st.session_state["verification_step"] = "select_role"
                                st.rerun()
                            else:
                                st.error(message)

                else:  # Recruiter
                    # Pre-verification workflow for recruiters
                    if st.session_state["verification_step"] == "select_role":
                        st.markdown("### Step 1: Request Access Code")
                        st.info("🔐 Recruiters must verify their email before registration.")
                        recruiter_email = st.text_input("Enter your email address", key="recruiter_email", autocomplete="email")

                        if st.button("Send Access Code"):
                            if not recruiter_email or not validate_email(recruiter_email):
                                st.error("Please enter a valid email address.")
                            elif not SMTP_USERNAME or not SMTP_PASSWORD:
                                st.error("Email service is not configured. Please contact administrator.")
                            else:
                                # Generate and save code
                                access_code = generate_access_code()
                                if save_invitation_code(recruiter_email, access_code):
                                    # Send email
                                    success, message = send_access_code_email(recruiter_email, access_code)
                                    if success:
                                        st.success("Access code sent to your email!")
                                        st.session_state["verification_email"] = recruiter_email
                                        st.session_state["verification_step"] = "verify_code"
                                        st.rerun()
                                    else:
                                        st.error(message)
                                else:
                                    st.error("Failed to generate access code. Please try again.")

                    elif st.session_state["verification_step"] == "verify_code":
                        st.markdown("### Step 2: Verify Access Code")
                        st.info(f"📧 Code sent to: {st.session_state['verification_email']}")
                        entered_code = st.text_input("Enter the 6-character access code", key="entered_code", max_chars=6)

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Verify Code"):
                                if not entered_code or len(entered_code) != 6:
                                    st.error("Please enter a valid 6-character code.")
                                elif verify_invitation_code(st.session_state["verification_email"], entered_code.upper()):
                                    st.success("✅ Code verified successfully!")
                                    st.session_state["verification_step"] = "complete_registration"
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid or expired code. Please try again.")

                        with col2:
                            if st.button("Resend Code"):
                                # Generate new code and resend
                                access_code = generate_access_code()
                                if save_invitation_code(st.session_state["verification_email"], access_code):
                                    success, message = send_access_code_email(st.session_state["verification_email"], access_code)
                                    if success:
                                        st.success("New access code sent!")
                                    else:
                                        st.error(message)
                                else:
                                    st.error("Failed to generate new code.")

                        if st.button("← Back to Email Input"):
                            st.session_state["verification_step"] = "select_role"
                            st.session_state["verification_email"] = ""
                            st.rerun()

                    elif st.session_state["verification_step"] == "complete_registration":
                        st.markdown("### Step 3: Complete Registration")
                        st.success(f"✅ Email verified: {st.session_state['verification_email']}")

                        name = st.text_input("Full Name", key="reg_name", autocomplete="name")
                        phone = st.text_input("Phone", key="reg_phone", autocomplete="tel")
                        password = st.text_input("Password", type="password", key="reg_password", autocomplete="new-password")
                        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm", autocomplete="new-password")

                        if st.button("Complete Registration"):
                            if password != confirm_password:
                                st.error("Passwords do not match.")
                            elif not all([name, phone, password]):
                                st.error("Please fill all fields.")
                            else:
                                success, message = register_user(name, st.session_state["verification_email"], phone, password, "recruiter")
                                if success:
                                    st.session_state["registration_message"] = "Recruiter registration completed successfully! Please login."
                                    # Reset session state
                                    st.session_state["verification_step"] = "select_role"
                                    st.session_state["verification_email"] = ""
                                    st.rerun()
                                else:
                                    st.error(message)

                        if st.button("← Start Over"):
                            st.session_state["verification_step"] = "select_role"
                            st.session_state["verification_email"] = ""
                            st.rerun()
    else:
        # Dashboard based on role
        user = st.session_state["user"]
        if user["role"] == "candidate":
            candidate_dashboard(user)
        elif user["role"] == "recruiter":
            recruiter_dashboard(user)
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state["user"] = None
            st.rerun()

        # About text below logout button
        st.sidebar.markdown("---")
        if user["role"] == "recruiter":
            st.sidebar.markdown(
                "**About**\n\n"
                "Recruiter dashboard: Create job descriptions, compute resume rankings, and shortlist top candidates efficiently. "
                "Use the system to compare applicant profiles, track matching scores, and speed up hiring decisions."
            )
        else:
            st.sidebar.markdown(
                "**About**\n\n"
                "Candidate dashboard: View your resume submission status and see how your profile aligns with the selected role. "
                "This system helps candidates understand hiring criteria and keep track of application progress."
            )


    st.markdown('</div>', unsafe_allow_html=True)
    # Footer
    st.markdown('<div class="footer">© 2026 Resume Screening System | Mekdela Amba University</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()