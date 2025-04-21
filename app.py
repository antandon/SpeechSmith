import streamlit as st
from pages.login_page import login_page
from pages.service_page_old import services

# Set page config must be first
st.set_page_config(
    page_title="SpeechSmith",
    page_icon="🎤",
    layout="wide"
)

def load_header_css():
    st.markdown("""
        <style>
        /* Navigation bar */
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: transparent;
            position: relative;
            z-index: 1000;
        }

        .nav-left {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-right {
            position: absolute;
            right: 2rem;
            top: 1rem;
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
        }

        .login-btn:hover {
            background: #ff3333 !important;
            transform: translateY(-2px);
        }

        /* Hero section */
        .hero-container {
            text-align: center;
            padding: 4rem 2rem;
            color: white;
            margin-top: -2rem;
        }

        .hero-title {
            font-size: 4rem;
            background: linear-gradient(135deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
        }

        .hero-subtitle {
            font-size: 2rem;
            color: white;
            margin-bottom: 2rem;
        }

        .read-more-btn {
            background: #6C63FF;
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            font-size: 1.2rem;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .read-more-btn:hover {
            background: #5a52cc;
            transform: translateY(-2px);
        }

        /* Background and general styling */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                        url("https://github.com/user-attachments/assets/9bc19a87-c89d-405e-94ca-7ad06a920e90") no-repeat center center fixed;
            background-size: cover;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Load header CSS
    load_header_css()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Navigation bar
    st.markdown("""
        <div class="nav-container">
            <div class="nav-left">
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
    
    # Hero section
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">SPEECHSMITH</h1>
            <h2 class="hero-subtitle">Elevate your speech. Amplify your impact!</h2>
            <button class="read-more-btn">Read More</button>
        </div>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.logged_in:
        if st.session_state.get('show_login', False):
            login_page()
    else:
        services()

if __name__ == "__main__":
    main() 