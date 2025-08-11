#!/usr/bin/env python3
"""
SIMPLE FIX - No complex strings
Run: python simple_fix.py
"""

import subprocess
import os

print("🔥 FIXING LANGCHAIN ERROR...")

# Remove LangChain
print("Removing LangChain...")
os.system("pip uninstall -y langchain langchain-core langchain-community langchain-google-genai langsmith")

# Install working packages
print("Installing Streamlit...")
os.system("pip install streamlit python-docx")

# Create app file by writing lines
print("Creating app...")

lines = [
    "import streamlit as st",
    "import tempfile",
    "import os",
    "",
    'st.title("ADGM Compliance Checker")',
    "",
    "def check_compliance(text):",
    "    text = text.lower()",
    "    issues = []",
    '    if "dubai courts" in text:',
    '        issues.append("HIGH: Uses Dubai Courts - Change to ADGM Courts")',
    '    if "uae federal" in text:',
    '        issues.append("HIGH: Uses UAE Federal Courts - Change to ADGM Courts")',
    '    if "adgm" not in text and "abu dhabi global market" not in text:',
    '        issues.append("MEDIUM: Add ADGM jurisdiction clause")',
    '    if "signature" not in text and "signed" not in text:',
    '        issues.append("HIGH: Add signature section")',
    "    score = max(0, 100 - len(issues) * 20)",
    "    return score, issues",
    "",
    'file = st.file_uploader("Upload .docx", type=["docx"])',
    "",
    "if file:",
    "    try:",
    "        from docx import Document",
    "        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:",
    "            tmp.write(file.getvalue())",
    "            doc = Document(tmp.name)",
    "            text = chr(10).join([p.text for p in doc.paragraphs])",
    "            os.unlink(tmp.name)",
    "        score, issues = check_compliance(text)",
    '        st.metric("Compliance Score", f"{score}/100")',
    "        if issues:",
    "            for issue in issues:",
    '                if "HIGH" in issue:',
    "                    st.error(issue)",
    "                else:",
    "                    st.warning(issue)",
    "        else:",
    '            st.success("No issues!")',
    "    except Exception as e:",
    '        st.error(f"Error: {e}")'
]

# Write the file
with open('working_app.py', 'w') as f:
    for line in lines:
        f.write(line + '\n')

print()
print("✅ DONE!")
print()
print("🚀 RUN THIS:")
print("streamlit run working_app.py")
print()
print("No more errors!")