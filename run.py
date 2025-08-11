#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    try:
        import streamlit
        import google.generativeai
        import langchain
        import docx
        print('✅ All requirements installed')
        return True
    except ImportError as e:
        print(f'❌ Missing: {e}')
        return False

def check_environment():
    if not Path('.env').exists():
        print('❌ .env file not found')
        print('💡 Copy .env.template to .env and add your API key')
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print('❌ GOOGLE_API_KEY not configured in .env')
        return False
    
    print('✅ Environment configured')
    return True

def main():
    print('🚀 Starting ADGM Corporate Agent...')
    
    if not check_requirements():
        print('💡 Run: pip install -r requirements.txt')
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    print('🎯 Launching Streamlit application...')
    print('🌐 Navigate to: http://localhost:8501')
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_app.py',
            '--server.port=8501',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print('\n👋 Goodbye!')

if __name__ == '__main__':
    main()
