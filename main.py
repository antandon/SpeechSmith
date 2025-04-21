import streamlit as st
import os
import json
from dotenv import load_dotenv
from pages.service_page import services
from pages.home_page import home
from pages.about_page import about
from pages.contact_page import contact

# Set page config must be first
st.set_page_config(
    page_title="SpeechSmith",
    page_icon="🎤",
    layout="wide"
)

# Load environment variables
load_dotenv()

def load_main_css():
    st.markdown("""
        <style>
        /* Main styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: #f8f5ff;
            font-size: 1.1rem;
        }
        
        /* Navigation styles */
        .nav {
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        .stButton > button {
            background: rgb(109, 40, 217);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 4px;
            border: none;
            font-size: 1rem;
            transition: background 0.3s ease;
        }

        .stButton > button:hover {
            background: rgb(91, 33, 182);
        }

        /* Login container */
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .login-title {
            text-align: center;
            color: rgb(109, 40, 217);
            margin-bottom: 2rem;
            font-size: 2rem;
        }

        .login-input {
            margin-bottom: 1rem;
        }

        .login-button {
            width: 100%;
            margin-top: 1rem;
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
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">Welcome Back!</h1>', unsafe_allow_html=True)
    
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        credentials = load_credentials()
        
        if email in credentials and credentials[email] == password:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.success("Login successful!")
            st.session_state.page = 'home'
            st.rerun()
        else:
            st.error("Invalid email or password")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load CSS
    load_main_css()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("Home"):
            st.session_state.page = 'home'
    with col2:
        if st.button("About"):
            st.session_state.page = 'about'
    with col3:
        if st.button("Services"):
            st.session_state.page = 'services'
    with col4:
        if st.button("Contact Us"):
            st.session_state.page = 'contact'
    with col5:
        if not st.session_state.logged_in:
            if st.button("Login"):
                st.session_state.page = 'login'
        else:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.email = None
                st.session_state.page = 'home'
                st.rerun()
    
    # Display the selected page
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'home':
        home()
    elif st.session_state.page == 'about':
        about()
    elif st.session_state.page == 'services':
        services()
    elif st.session_state.page == 'contact':
        contact()

if __name__ == "__main__":
    main()