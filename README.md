# ADGM Corporate Agent 🏛️⚖️

**AI-Powered Legal Document Intelligence for Abu Dhabi Global Market Compliance**

## 🚀 Quick Start (Windows)

### 1. Setup & Install
```powershell
# Run setup script
.\setup_windows.ps1

# Or install manually
pip install -r requirements.txt
```

### 2. Configure Environment
```powershell
# Copy template and edit
copy .env.template .env
# Edit .env: GOOGLE_API_KEY=your_actual_key_here
```

### 3. Launch Application
```powershell
# Generate test documents
python create_test_documents.py

# Run the app
python run.py
```

## 🎯 Features

- **📄 Document Intelligence**: Parse and analyze .docx legal documents
- **⚖️ ADGM Compliance**: Check against official ADGM regulations
- **🚩 Red Flag Detection**: Identify legal issues and compliance gaps
- **📝 Smart Commenting**: Add contextual comments to documents
- **📊 Professional Dashboard**: Interactive Streamlit interface
- **📥 Export Capabilities**: Download reviewed documents and reports

## 🧪 Testing

Upload the generated test documents:
- `articles_with_issues.docx` - Contains multiple compliance violations
- `compliant_articles.docx` - Fully ADGM compliant example

## 📋 Windows Requirements

- Python 3.8+ (from python.org)
- Google Gemini API key
- PowerShell 5.1+
- Modern web browser

## 🔧 Troubleshooting

- **PowerShell Execution Policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Python not found**: Install from python.org and restart terminal
- **API errors**: Check your GOOGLE_API_KEY in .env file
- **Dependencies**: Run `pip install -r requirements.txt` again

---

Built for **2Cents Capital AI Engineer Intern Challenge** 🏆
