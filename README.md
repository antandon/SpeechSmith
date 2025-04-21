# SpeechSmith Project

A speech analysis and refinement application built with Streamlit.

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   DEEPGRAM_API_KEY=your_deepgram_key
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```
5. Run the application:
   ```bash
   streamlit run pages/service_page.py
   ```

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Create a GitHub repository for your project
2. Push your code to GitHub
3. Go to [Streamlit Cloud](https://streamlit.io/cloud)
4. Sign in with your GitHub account
5. Click "New app"
6. Select your repository and branch
7. Set the main file path to `pages/service_page.py`
8. Add your environment variables in the "Advanced settings"
9. Click "Deploy"

### Option 2: Heroku

1. Install the Heroku CLI
2. Create a `Procfile`:
   ```
   web: streamlit run pages/service_page.py --server.port $PORT
   ```
3. Create a `runtime.txt`:
   ```
   python-3.9.13
   ```
4. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set GROQ_API_KEY=your_key
   heroku config:set DEEPGRAM_API_KEY=your_key
   heroku config:set HUGGINGFACE_TOKEN=your_token
   ```

### Option 3: AWS Elastic Beanstalk

1. Install AWS CLI and EB CLI
2. Initialize your project:
   ```bash
   eb init -p python-3.9 speechsmith
   ```
3. Create an environment:
   ```bash
   eb create speechsmith-env
   ```
4. Set environment variables:
   ```bash
   eb setenv OPENAI_API_KEY=your_key GROQ_API_KEY=your_key DEEPGRAM_API_KEY=your_key HUGGINGFACE_TOKEN=your_token
   ```
5. Deploy:
   ```bash
   eb deploy
   ```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `GROQ_API_KEY`: Your Groq API key
- `DEEPGRAM_API_KEY`: Your Deepgram API key
- `HUGGINGFACE_TOKEN`: Your Hugging Face token

## Notes

- Make sure to install ffmpeg for audio processing
- The application requires significant memory for ML models
- Consider using a GPU instance for better performance 