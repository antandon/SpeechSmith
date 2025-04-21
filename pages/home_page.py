import streamlit as st
from pages.service_page import services

def load_home_css():
    st.markdown("""
        <style>
        /* Full background image setup */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(255, 255, 255, 0.2)),
                        url("https://github.com/user-attachments/assets/4b645ecc-65e9-4589-a980-5ed2b855df5f") no-repeat center center fixed;
            background-size: cover;
        }
       

        /* Hero section styling */
        .hero {
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(120, 116, 116,0.2);
            border-radius: 12px;
            max-width: 800px;
            margin: 0 auto;
            box-shadow: 0 4px 8px rgba(142, 138, 138, 0.2);
            
        }
        
        .hero h1 {
            font-size: 3.5rem;
            background: linear-gradient(45deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 1rem;
        }
        
        .hero p {
            font-size: 1.5rem;
            color: #f0f0f0;
        }

        /* Hide default Streamlit button styling */
        .stButton>button {
            background: #6C63FF;
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.2rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: auto;
            margin: 0 auto;
            display: block;
        }

        
        /* Additional content styling */
        .additional-content {
            background: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
            display: none;
        }
        
        .additional-content.visible {
            display: block;
        }
        
        /* Center container */
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def home():
    load_home_css()
    
    # Create session state to track if additional content should be shown
    if 'show_more' not in st.session_state:
        st.session_state.show_more = False
    
    # Hero section
    st.markdown("""
        <div class="hero">
            <h1>SPEECHSMITH</h1>
            <p>Elevate your speech. Amplify your impact!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the button using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Read More"):
            st.session_state.show_more = not st.session_state.show_more
    
    # Additional content that appears when button is clicked
    if st.session_state.show_more:
        st.markdown("""
                    <div class="additional-content visible">
        <h2>Welcome to SpeechSmith</h2>
        <p>Welcome to SpeechSmith, where we aim to help individuals across India unlock the art of powerful public speaking. Whether you’re a student preparing for a debate, a young professional gearing up for a presentation, or simply looking to become a more fluent and articulate speaker, SpeechSmith offers a unique platform to elevate your speech skills with tailored feedback and structured improvement paths.</p>

        <h3>Our Mission</h3>
        <p>At SpeechSmith, we believe that effective communication is key to success in every field. Our mission is to empower young Indians by honing their public speaking skills, boosting their confidence, and preparing them to deliver speeches with impact. From pronunciation and articulation to structure and pace, we help you polish every aspect of your speech.</p>

        <h3>How It Works</h3>
        <p>Simply upload your speech as an audio or video file, answer a few questions about your purpose and audience, and receive detailed, AI-driven feedback on various parameters such as pronunciation accuracy, clarity, speech rate, and more. With SpeechSmith, improving your public speaking skills is just a few clicks away!</p>

        <h3>Get Started</h3>
        <p>Ready to become a more effective communicator? Click the “Get Started” button below to upload your first speech and take the first step towards mastering the art of public speaking!</p>
        
        </div>
    """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            .get-started-button > button {
                background-color: #800080;  /* Purple background */
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 4px;
                border: none;
                font-size: 1rem;
                transition: background 0.3s ease;
            }
        
            .get-started-button > button:hover {
                background-color: #6a006a;  /* Darker purple on hover */
            }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            if st.button("Get Started",key="get_started_button"):
                st.session_state.page = 'services'
                

    st.markdown("""
        <style>
            .get-started-button {
                display: inline-block;
                padding: 10px 20px;
                margin-top: 20px;
                font-size: 16px;
                font-color: 
                color: white;
                background-color: #800080;
                border-radius: 5px;
                text-decoration: none;
            }

            .get-started-button:hover {
                background-color: #800080;
            }
        </style>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    home()
    