import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from groq import Groq
import torch
from transformers import pipeline
import wave
import contextlib
from textblob import TextBlob
import speech_recognition as sr
import tempfile
import os
import json
import re
from dotenv import load_dotenv
import markdown
from typing import Optional
load_dotenv()

class SpeechAnalyzer:
    def __init__(self):
        # Initialize Groq client
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
    
    def get_audio_duration(self, audio_path):
        """Get duration of audio file"""
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration
    
    def transcribe_audio(self, audio_path):
        """Convert speech to text"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            st.error(f"Error in transcription: {str(e)}")
            return None

    def analyze_text_with_llama(self, text, analysis_type, topic=None ):
        """Analyze text using Llama 3 model via Groq"""
        prompts = {
            'pronunciation': f"""
                Analyze the following text to identify words that were mispronounced during speech:
                
                Text: {text}
                
                Please provide:
                1. List of words that were mispronounced
                2. Confidence score for each identified mispronunciation (0-1, where 1 indicates high confidence that the word was mispronounced)
                3. Overall speech accuracy assessment
                4. Correct pronunciation guidance for the mispronounced words
                
                Return the response as a JSON string with keys: 'mispronounced_words', 'confidence_scores', 'accuracy_assessment', 'pronunciation_guidance'
            """,
            'mood': f"""
                Analyze the mood and emotional content of the following text:
                Text: {text}
                Topic: {topic}
                
                Please provide:
                1. Primary emotion
                2. Secondary emotions
                3. Emotional intensity (0-1)
                4. Formality level
                5. Target audience suitability
                6. Mood suitability assessment:
                   - Evaluate if the identified mood aligns with the speech topic
                   - Provide specific reasons why the mood is appropriate or needs adjustment
                
                Return the response as a JSON string with keys: 'primary_emotion', 'secondary_emotions', 
                'intensity', 'formality', 'audience_suitability', 'mood_suitability_assessment'
            """,
            'filler_analysis': f"""
                Analyze the following text for filler words and speech patterns:
                Text: {text}
                Please provide:
                1. List of filler words found
                2. Count of each filler word
                3. Total word count
                4. Speaking style assessment
                5. Detailed suggestions for improvement
                Return the response as a JSON string with keys: 'filler_words', 'counts', 'total_words', 
                'style_assessment', 'suggestions'
            """
        }

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional speech analyst. Provide detailed analysis in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompts[analysis_type]
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=3000,
            )
            
            # Extract JSON from response
            response_text = completion.choices[0].message.content
            # Find JSON string within the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
                
        except Exception as e:
            st.error(f"Error in Llama analysis: {str(e)}")
            return None

    def analyze_pronunciation(self, audio_path,transcript):
        """Analyze pronunciation using audio features and Llama"""
        # Get basic audio features
        y, sr = librosa.load(audio_path)
        
        # Get Llama analysis for pronunciation
        llama_analysis = self.analyze_text_with_llama(transcript, 'pronunciation')

        if llama_analysis:
            return {
                'confidence_scores': llama_analysis['confidence_scores'],
                'mispronounced_words': llama_analysis['mispronounced_words'],
                'accuracy_assessment': llama_analysis['accuracy_assessment'],
                'pronunciation_guidance': llama_analysis.get('pronunciation_guidance', {})
            }
        else:
            return None

    def analyze_pitch(self, audio_path, gender):
        """Analyze pitch characteristics"""
        if gender not in ['male', 'female']:
            return None
        
        y, sr = librosa.load(audio_path)
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get non-zero pitches
        pitches_flat = pitches[magnitudes > np.max(magnitudes) * 0.1]
        pitches_flat = pitches_flat[pitches_flat > 0]
        
        if len(pitches_flat) == 0:
            return {
                'monotonous': True,
                'erratic': False,
                'mean_pitch': 0,
                'pitch_std': 0
            }
        
        pitch_mean = np.mean(pitches_flat)
        pitch_std = np.std(pitches_flat)
        
        thresholds = {
            'male': {'monotonous': 20, 'erratic': 80},
            'female': {'monotonous': 30, 'erratic': 100}
        }
        
        current_threshold = thresholds[gender]
        
        return {
            'monotonous': bool(pitch_std < current_threshold['monotonous']),
            'erratic': bool(pitch_std > current_threshold['erratic']),
            'mean_pitch': float(pitch_mean),
            'pitch_std': float(pitch_std)
        }

    def analyze_speech_rate(self, audio_path, transcript):
        """Analyze speech rate and pauses"""
        duration = self.get_audio_duration(audio_path)
        y, sr = librosa.load(audio_path)
        
        intervals = librosa.effects.split(y, top_db=20)
        speech_duration = sum(interval[1] - interval[0] for interval in intervals) / sr
        
        filler_analysis = self.analyze_text_with_llama(transcript, 'filler_analysis')
        
        total_words = len(transcript.split())
        filler_count = sum(filler_analysis['counts'].values()) if filler_analysis else 0
        effective_words = total_words - filler_count
        
        # Calculate words per minute
        words_per_minute = (effective_words / duration) * 60
        
        return {
            'speech_rate': words_per_minute,
            'balance_ratio': speech_duration / duration,
            'word_count': effective_words,
            'total_duration': duration,
            'filler_analysis': filler_analysis,
            'total_words': total_words,
            'filler_count': filler_count
        }

    def analyze_mood(self, transcript, topic):
        """Analyze speech mood using Llama"""
        mood_analysis = self.analyze_text_with_llama(transcript,  'mood', topic)
        return mood_analysis

def analyze_speaking_style(rate_analysis):
    """Analyze speaking style based on rate and filler words"""
    wpm = rate_analysis['speech_rate']
    balance = rate_analysis['balance_ratio']
    filler_ratio = rate_analysis['filler_count'] / rate_analysis['total_words'] if rate_analysis['total_words'] > 0 else 0
    
    style_feedback = []
    
    # Speech rate analysis
    if wpm < 120:
        style_feedback.append("Speaking rate is slow (below 120 words per minute). Consider increasing your pace.")
    elif wpm > 160:
        style_feedback.append("Speaking rate is fast (above 160 words per minute). Consider slowing down.")
    else:
        style_feedback.append("Speaking rate is at a good pace (120-160 words per minute).")
    
    # Balance ratio analysis
    if balance < 0.65:
        style_feedback.append("Too many pauses detected. Try to maintain a more continuous flow.")
    elif balance > 0.9:
        style_feedback.append("Very few pauses detected. Consider adding strategic pauses for emphasis.")
    else:
        style_feedback.append("Good balance between speech and pauses.")
    
    # Filler words analysis
    if filler_ratio > 0.1:
        style_feedback.append(f"High usage of filler words ({filler_ratio:.1%}). Work on reducing them.")
    else:
        style_feedback.append(f"Good control over filler words ({filler_ratio:.1%}).")
    
    # Word count analysis
    duration_minutes = rate_analysis['total_duration'] / 60
    ideal_word_count = int(duration_minutes * 140)  # assuming 140 wpm is ideal
    actual_words = rate_analysis['total_words']
    
    if actual_words < ideal_word_count * 0.8:
        style_feedback.append(f"Content is too brief for the duration. Consider adding more content (ideal: {ideal_word_count} words).")
    elif actual_words > ideal_word_count * 1.2:
        style_feedback.append(f"Content is too long for the duration. Consider reducing content (ideal: {ideal_word_count} words).")
    else:
        style_feedback.append(f"Good content length for the duration ({actual_words} words).")
    
    return style_feedback


def generate_feedback(analyzer_results, topic):
    """Generate comprehensive feedback with improved formatting"""
    feedback = {
        'pronunciation': [],
        'mood': [],
        'speaking style': [],
        'pitch': []
    }
    
    # Pronunciation feedback
    if analyzer_results.get('pronunciation'):
        pron = analyzer_results['pronunciation']
        feedback['pronunciation'].extend([
            {'type': 'main', 'content': f"Overall speech accuracy: {pron['accuracy_assessment']}"},
            {'type': 'header', 'content': "Mispronounced words and confidence scores:"}
        ])
        for word, score in pron['confidence_scores'].items():
            feedback['pronunciation'].append(
                {'type': 'sub', 'content': f"{word}: {score:.2f} "}
            )
        if pron.get('pronunciation_guidance'):
            feedback['pronunciation'].append(
                {'type': 'header', 'content': "Correct pronunciation guide:"}
            )
            for word, guidance in pron['pronunciation_guidance'].items():
                feedback['pronunciation'].append(
                    {'type': 'sub', 'content': f"{word}: {guidance}"}
            )

    # Mood feedback
    if analyzer_results.get('mood'):
        mood = analyzer_results['mood']
        feedback['mood'].extend([
            {'type': 'main', 'content': f"Primary emotion: {mood['primary_emotion']}"},
            {'type': 'main', 'content': f"Formality level: {mood['formality']}"},
            {'type': 'main', 'content': f"Audience suitability: {mood['audience_suitability']}"}
        ])
        if isinstance(mood['mood_suitability_assessment'], dict):
            feedback['mood'].append(
                {'type': 'main', 'content': f"Mood suitability: {mood['mood_suitability_assessment']['alignment']}"},
            )
            feedback['mood'].append(
                {'type': 'sub', 'content': f"Reasons: {mood['mood_suitability_assessment']['reasons']}"}
            )
        else:
            feedback['mood'].append(
                {'type': 'main', 'content': f"Mood suitability: {mood['mood_suitability_assessment']}"}
            )

    # Speaking style feedback
    if analyzer_results.get('speech_rate'):
        style_feedback = analyze_speaking_style(analyzer_results['speech_rate'])
        feedback['speaking style'] = [
            {'type': 'main', 'content': item} for item in style_feedback
        ]

    # Pitch feedback
    if analyzer_results.get('pitch'):
        pitch = analyzer_results['pitch']
        if pitch:
            feedback['pitch'].extend([
                {'type': 'header', 'content': "Pitch Analysis:"},
                {'type': 'sub', 'content': 'Voice is monotonous. Consider adding more pitch variation.' if pitch['monotonous'] else 'Good pitch variation in voice.'},
                {'type': 'sub', 'content': 'Pitch variation is erratic. Try to maintain more consistent variation.' if pitch['erratic'] else 'Consistent pitch variation.'},
                {'type': 'sub', 'content': f"Average pitch: {pitch['mean_pitch']:.1f} Hz"},
                {'type': 'sub', 'content': f"Pitch variation: {pitch['pitch_std']:.1f} Hz"}
            ])
    
    return feedback

def format_feedback_to_html(feedback, transcription, refined_speech):
    def format_list_items(items):
        if not items:
            return "<p>No feedback available</p>"
        
        html = ""
        for item in items:
            if item['type'] == 'header':
                html += f"<p class='feedback-header'>• {item['content']}</p>"
            elif item['type'] == 'main':
                html += f"<p class='feedback-main'>• {item['content']}</p>"
            elif item['type'] == 'sub':
                html += f"<p class='feedback-sub'>      ◦ {item['content']}</p>"
        return html

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])
    
    # Convert refined speech markdown to HTML
    refined_speech_html = md.convert(refined_speech) if refined_speech else ""
    print(refined_speech_html)
    # Create HTML for each feedback section
    feedback_sections = ""
    for key in feedback:
        feedback_sections += f"""
            <div class="feedback-section">
                <h4 class="feedback-title">{key.title()}</h4>
                {format_list_items(feedback[key])}
            </div>
        """
    
    # Combine everything into the main content template with CSS
    content = f"""
        
        <div class="content-section">
            <h3>Original Transcription</h3>
            <p>{transcription}</p>
            
            <h3>Refined Speech</h3>
            <p>{refined_speech_html}</p>
            
            <h3>Detailed Feedback</h3>
            <div class="feedback-details">
                {feedback_sections}
            </div>
        </div>
    """
    return content

def main():
    st.title("Advanced Speech Analysis Tool")
    
    # Initialize analyzer
    analyzer = SpeechAnalyzer()
    
    # User inputs
    topic = st.text_input("Topic of Speech")
    audio_file = st.file_uploader("Upload Audio File (.wav format)", type=['wav'])
    gender = st.radio("Select Gender (for pitch analysis)", ['male', 'female'])
    
    if audio_file and topic:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        try:
            # Process audio
            with st.spinner("Analyzing speech..."):
                # Get transcript
                transcript = analyzer.transcribe_audio(audio_path)
                
                if transcript:
                    # Perform analysis
                    results = {
                        'pronunciation': analyzer.analyze_pronunciation(audio_path, transcript),
                        'pitch': analyzer.analyze_pitch(audio_path, gender),
                        'speech_rate': analyzer.analyze_speech_rate(audio_path, transcript),
                        'mood': analyzer.analyze_mood(transcript)
                    }
                    
                    # Generate feedback
                    feedback = generate_feedback(results,topic)
                    
                    # Display results
                    st.header("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Transcript")
                        st.write(transcript)
                        
                        st.subheader("Pronunciation Analysis")
                        if results['pronunciation']:
                            st.json(results['pronunciation'])
                            st.write(results["pronunciation"]["difficult_words"])
                    
                    with col2:
                        st.subheader("Mood Analysis")
                        if results['mood']:
                            st.json(results['mood'])
                        
                        st.subheader("Speech Characteristics")
                        st.write(f"Speech Rate: {results['speech_rate']['speech_rate']:.2f} words/second")
                        # st.write(f"")
                        st.write(f"Balance Ratio: {results['speech_rate']['balance_ratio']:.2f}")
                    
                        st.subheader("Pitch Results")
                        st.write(results['pitch'])

                        st.subheader("Full Results")
                        st.write(results)
                    # Detailed feedback
                    st.header("Detailed Feedback")
                    for category, items in feedback.items():
                        if items:
                            st.subheader(category.title())
                            for item in items:
                                st.write(f"- {item}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(audio_path)

if __name__ == "__main__":
    main()