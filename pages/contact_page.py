# pages/contact_page.py
import streamlit as st
from PIL import Image
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_contact_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(255, 255, 255, 0.2)),
                        url("https://github.com/user-attachments/assets/bc0cf4c2-5598-4dbf-8aa3-a2029a516f8e") no-repeat center center fixed;
            background-size: cover;
        }
        
        .contact-container {
            max-width: 1200px;
            width: 100%;
            padding: 3rem;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
            position: relative;
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 600px;  /* Increased height */
            background: none;  /* Removed white background */
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);  /* Stronger shadow for a modern feel */
            overflow: hidden;
        }

        .image-container img {
            border-radius: 16px;
            width: 100%;
            height: 100%;
            object-fit: cover;  /* Ensures the image covers the entire container without distortion */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .image-container img:hover {
            transform: scale(1.05);  /* Subtle zoom effect on hover */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);  /* Enhanced shadow on hover */
        }
            
        .gradient-text {
            background: linear-gradient(45deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .contact-footer {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 2rem;
            display: flex;
            justify-content: space-around;
            align-items: center;
            flex-wrap: wrap;
        }
        .contact-info-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #4a5568;
            margin: 0.5rem;
        }
        .contact-info-item svg {
            width: 24px;
            height: 24px;
        }
        @media (max-width: 768px) {
            .contact-footer {
                flex-direction: column;
                gap: 1rem;
            }
        }
        
        .success-message {
            background: #4CAF50;
            color: white;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            text-align: center;
        }
        .stTextInput, .stTextArea {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .contact-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .contact-title {
            text-align: center;
            color: rgb(109, 40, 217);
            margin-bottom: 2rem;
            font-size: 2.5rem;
        }
        
        .contact-subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.2rem;
        }
        
        .stTextInput > div > div > input {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 0.75rem;
            border-radius: 4px;
            font-size: 1rem;
        }
        
        .stTextArea > div > div > textarea {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 0.75rem;
            border-radius: 4px;
            font-size: 1rem;
            min-height: 150px;
        }
        
        .contact-form {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def contact():
    load_contact_css()
    
    # Container for the entire contact section
    st.markdown('<h2 class="gradient-text">Contact Us</h2>', unsafe_allow_html=True)
    
    # Create two columns: one for the form and one for the image
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Contact Form
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Name", key="name", placeholder="Your name")
            email = st.text_input("Email ID", key="email", placeholder="Your email")
            subject = st.text_input("Subject", key="subject", placeholder="Subject of your message")
            message = st.text_area("Message", key="message", placeholder="Your message", height=150)
            
            submitted = st.form_submit_button("Send Message")
            
            if submitted:
                if name and email and subject and message:
                    try:
                        # Get email credentials from environment variables
                        sender_email = os.getenv('GMAIL_ADDRESS')
                        app_password = os.getenv('GMAIL_APP_PASSWORD')
                        
                        # Create message
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = sender_email  # Sending to the same address
                        msg['Subject'] = f"Contact Form: {subject}"

                        # Email body
                        body = f"""
                        New contact form submission:
                        
                        Name: {name}
                        Email: {email}
                        Subject: {subject}
                        
                        Message:
                        {message}
                        """
                        msg.attach(MIMEText(body, 'plain'))

                        # Send email
                        with smtplib.SMTP('smtp.gmail.com', 587) as server:
                            server.starttls()
                            server.login(sender_email, app_password)
                            server.send_message(msg)
                            
                            # Only clear form and show success if email is sent successfully
                            st.success("Thank you for your message! We'll get back to you soon.")
                            return  # Exit the function after successful submission
                            
                    except Exception as e:
                        st.error(f"There was an error sending your message: {str(e)}")
                else:
                    st.warning("Please fill in all fields.")
    
    with col2:
        # Try to load and display the contact image
        html_code = """
        <div class="image-container">
            <img src="https://github.com/user-attachments/assets/03d06851-6c6e-48d6-b977-22c58cbd6922" alt="Image not found">
        </div>
        """

        # Rendering the image with the specified HTML and CSS
        st.markdown(html_code, unsafe_allow_html=True)
    
    # Footer with contact information
    st.markdown("""
        <div class="contact-footer">
            <div class="contact-info-item">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" 
                     stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                    <polyline points="22,6 12,13 2,6"/>
                </svg>
                <span>speechsmith.ai@gmail.com</span>
            </div>
            <div class="contact-info-item">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" 
                     stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"/>
                </svg>
                <span>Get in touch with us!</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    contact()
