import streamlit as st
import os
import json
from dotenv import load_dotenv

# Set page config must be first
st.set_page_config(
    page_title="SpeechSmith - Login",
    page_icon="🎤",
    layout="wide"
)

# Load environment variables
load_dotenv()

def load_services_css():
    st.markdown("""
        <style>
        /* Navigation bar */
        .nav-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            background: transparent;
            position: relative;
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }

        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
            margin-right: auto;
            margin-left: 20%;
        }

        .nav-right {
            position: absolute;
            right: 2rem;
        }

        .nav-link {
            background: #6C63FF;
            color: white;
            text-decoration: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            display: inline-block;
            font-family: system-ui, -apple-system, sans-serif;
        }

        .nav-link:hover {
            background: #5a52cc;
            transform: translateY(-2px);
        }

        .login-btn {
            background: #FF4B4B !important;
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            font-family: system-ui, -apple-system, sans-serif;
        }

        .login-btn:hover {
            background: #ff3333 !important;
            transform: translateY(-2px);
        }

        /* Login container */
        .login-container {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 500px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                        0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Input fields */
        .stTextInput>div>div>input {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        /* Login button */
        .stButton>button {
            background: #FF4B4B !important;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.8rem 1.5rem;
            width: 100%;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background: #ff3333 !important;
            transform: translateY(-2px);
        }

        /* Background and general styling */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                        url("https://github.com/user-attachments/assets/9bc19a87-c89d-405e-94ca-7ad06a920e90") no-repeat center center fixed;
            background-size: cover;
        }

        /* Hero title */
        .hero-title {
            font-size: 3rem;
            background: linear-gradient(135deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            margin-top: 2rem;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def load_credentials():
    """Load credentials from a JSON file"""
    try:
        with open('credentials.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Create default credentials if file doesn't exist
        credentials = {}
        for i in range(1, 41):
            email = f"speechsmith{i}@example.com"
            password = f"speechsmith{i}"
            credentials[email] = password
        
        # Save credentials to file
        with open('credentials.json', 'w') as f:
            json.dump(credentials, f)
        return credentials

def login_page():
    # Load CSS
    load_services_css()
    
    # Navigation bar
    st.markdown("""
        <div class="nav-container">
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/About" class="nav-link">About</a>
                <a href="/Services" class="nav-link">Services</a>
                <a href="/Contact_Us" class="nav-link">Contact Us</a>
            </div>
            <div class="nav-right">
                <a href="/login" class="login-btn">Login</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="hero-title">Welcome Back!</h1>', unsafe_allow_html=True)
    
    # Login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Login form
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        credentials = load_credentials()
        
        if email in credentials and credentials[email] == password:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.success("Login successful!")
            # Redirect to home page
            st.switch_page("app.py")
        else:
            st.error("Invalid email or password")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: white;">
            <p>© 2024 SpeechSmith. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    login_page() 