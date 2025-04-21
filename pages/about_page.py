import streamlit as st

def load_about_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(255, 255, 255, 0.2)),
                        url("https://github.com/user-attachments/assets/3266eaca-7c2f-4401-88e6-9e01e7e337d0") no-repeat center center fixed;
            background-size: cover;
        }

        .about-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 3rem;
            font-family: 'Inter', sans-serif;
        }

        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 400px;
            height: 450px;
            background: none;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        .image-container img {
            border-radius: 16px;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .image-container img:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        }

        .gradient-text {
            background: linear-gradient(135deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
            letter-spacing: -0.02em;
        }

        .content-section {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                        0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .content-section h3 {
            font-size: 1.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        .section-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            text-align: center;
            margin-bottom: 3rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        .content-section li {
            color: #334155;
            font-size: 1.1rem;
            line-height: 1.75;
            margin-bottom: 0.75rem;
        }

        ul {
            color: rgba(255, 255, 255, 0.85);
            margin-left: 1.5rem;
            margin-bottom: 2rem;
        }

        strong {
            color: #800080;
            font-weight: 600;
        }

        /* New styles for the flex layout */
        .content-wrapper {
            display: flex;
            gap: 2rem;
            align-items: flex-start;
            margin-top: 2rem;
        }

        .content-text {
            flex: 3;
        }

        .content-images {
            flex: 2;
            position: sticky;
            top: 2rem;
        }

        @media (max-width: 768px) {
            .content-wrapper {
                flex-direction: column;
            }
            .content-images {
                position: static;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def about():
    load_about_css()

    content = """
        <div class="about-container">
            <div class="gradient-text">
                About SpeechSmith
            </div>
            
            <p class="section-subtitle">
                At SpeechSmith, we are dedicated to helping you craft and deliver speeches that resonate. 
                Our innovative platform combines cutting-edge speech analysis with personalized coaching 
                to refine every aspect of your delivery.
            </p>

            <div class="content-wrapper">
                <div class="content-text">
                    <div class="content-section">
                        <h3>Our Vision</h3>
                        <p>SpeechSmith was created with a vision to bridge the communication gap for young people across India by making public speaking skills accessible and achievable. We believe that confident communication has the power to open doors and create opportunities, especially for students preparing for public speaking or debates. Our platform combines advanced technology with personalized feedback to help you develop a natural, fluent speaking style.</p>

                        <h3>Who Can Use SpeechSmith?</h3>
                        <ul>
                            <li><strong>Students:</strong> Tailored for students preparing for speeches, debates, or school presentations. SpeechSmith offers an effective way to improve your delivery and boost your confidence.</li>
                            <li><strong>Young Professionals and Interns:</strong> Early-career professionals can also benefit from SpeechSmith's feedback, whether preparing for team meetings, client presentations, or networking events.</li>
                            <li><strong>Public Speaking Enthusiasts:</strong> Anyone passionate about becoming a better communicator can use our platform to refine their skills and enhance their articulation and fluency.</li>
                        </ul>

                        <h3>Our Technology and Approach</h3>
                        <p>SpeechSmith uses state-of-the-art AI tools to analyze various elements of your speech, including pronunciation, semantic accuracy, articulation rate, and filler words. Our focus is on helping you polish each component of your delivery to ensure clarity, impact, and a professional edge.</p>

                        <h3>Empowering India's Future Speakers</h3>
                        <p>Our platform is designed to nurture confident, articulate speakers across India, preparing the next generation of communicators for success in the classroom, boardroom, and beyond.</p>
                    </div>
                </div>

                <div class="content-images">
                    <div class="image-container">
                        <img src="https://github.com/user-attachments/assets/c95ab315-7ee1-4b8d-b053-8fac129762e5" alt="Image not found">
                    </div>
                    <div class="image-container">
                        <img src="https://github.com/user-attachments/assets/1b54ebf3-5a0a-4150-aaa9-7b199ed16a14" alt="Image not found">
                    </div>
                </div>
            </div>
        </div>
    """
    st.html(content)