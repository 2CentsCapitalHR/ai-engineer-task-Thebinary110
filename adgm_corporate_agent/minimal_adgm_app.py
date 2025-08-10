import streamlit as st
import os
import json
import openai
from pathlib import Path
import io
from docx import Document as DocxDocument

st.set_page_config(
    page_title="ADGM Corporate Agent",
    page_icon="🏛️",
    layout="wide"
)

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 20px; border-radius: 10px; text-align: center;">
    <h1>🏛️ ADGM Corporate Agent</h1>
    <p>AI-Powered Legal Document Assistant</p>
    <p><em>Now with REAL AI Analysis!</em></p>
</div>
""", unsafe_allow_html=True)

def extract_docx_text(file_content):
    """Extract text from DOCX file."""
    try:
        doc = DocxDocument(io.BytesIO(file_content))
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        return "\n".join(text)
    except:
        return ""

def analyze_document_with_ai(text_content, api_key):
    """Analyze document using OpenAI."""
    
    if not text_content.strip():
        return {
            "document_type": "Empty Document",
            "compliance_issues": ["Document appears to be empty"],
            "risk_level": "High",
            "suggestions": ["Please upload a valid document with content"]
        }
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
You are an expert ADGM legal compliance analyst. Analyze this document for ADGM compliance.

Document content:
{text_content[:2000]}

Provide analysis in this JSON format:
{{
    "document_type": "specific document type (e.g., Articles of Association, Board Resolution, etc.)",
    "compliance_issues": ["list of specific compliance issues found"],
    "risk_level": "Low|Medium|High|Critical",
    "adgm_compliant": true/false,
    "suggestions": ["specific actionable suggestions"],
    "key_findings": ["important observations about the document"]
}}

Focus on:
1. ADGM-specific requirements
2. Jurisdiction clauses (should specify ADGM Courts)
3. Required sections and clauses
4. Legal language clarity
5. Signature requirements

Respond with valid JSON only.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert ADGM legal compliance analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response
        if result.startswith("```json"):
            result = result[7:-3]
        elif result.startswith("```"):
            result = result[3:-3]
        
        return json.loads(result)
        
    except json.JSONDecodeError:
        return {
            "document_type": "Analysis Error",
            "compliance_issues": ["Failed to parse AI analysis"],
            "risk_level": "Medium",
            "adgm_compliant": False,
            "suggestions": ["Please try again or check document format"],
            "key_findings": ["AI analysis returned invalid format"]
        }
    except Exception as e:
        return {
            "document_type": "Processing Error",
            "compliance_issues": [f"Analysis failed: {str(e)}"],
            "risk_level": "Medium", 
            "adgm_compliant": False,
            "suggestions": ["Check API key and try again"],
            "key_findings": ["Document processing encountered an error"]
        }

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Enhanced Features")
    st.markdown("""
    - 🤖 **Real AI Analysis**
    - 🔍 ADGM Compliance Check  
    - ⚠️ Risk Assessment
    - 💡 Smart Suggestions
    - 📊 Detailed Reporting
    """)
    
    st.markdown("## 📋 Supported Documents")
    supported_docs = [
        "Articles of Association",
        "Memorandum of Association", 
        "Board/Shareholder Resolutions",
        "Incorporation Applications",
        "UBO Declarations",
        "Register of Members/Directors"
    ]
    
    for doc in supported_docs:
        st.markdown(f"• {doc}")
    
    st.markdown("## 🔑 AI Analysis")
    st.info("Uses OpenAI GPT-3.5 for real document analysis")

# Main area
st.markdown("## 📤 Document Upload & AI Analysis")

# API Key input
api_key = st.text_input(
    "🔑 OpenAI API Key", 
    type="password", 
    help="Enter your OpenAI API key for real AI analysis"
)

uploaded_files = st.file_uploader(
    "Upload ADGM Documents (.docx)",
    type=['docx'],
    accept_multiple_files=True,
    help="Upload one or more DOCX documents for AI-powered ADGM compliance analysis"
)

if st.button("🚀 Analyze with AI", type="primary"):
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one document")
    else:
        # Process each document
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}", expanded=True):
                
                # Show processing status
                with st.spinner(f"🤖 AI analyzing {file.name}..."):
                    
                    # Extract document content
                    file_content = file.read()
                    text_content = extract_docx_text(file_content)
                    
                    # Perform AI analysis
                    ai_analysis = analyze_document_with_ai(text_content, api_key)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📄 File Size", f"{len(file_content):,} bytes")
                    st.metric("📝 Word Count", len(text_content.split()) if text_content else 0)
                    st.metric("📋 Document Type", ai_analysis.get("document_type", "Unknown"))
                
                with col2:
                    risk_level = ai_analysis.get("risk_level", "Medium")
                    risk_color = {
                        "Low": "🟢", "Medium": "🟡", 
                        "High": "🟠", "Critical": "🔴"
                    }.get(risk_level, "🟡")
                    
                    st.metric("⚠️ Risk Level", f"{risk_color} {risk_level}")
                    
                    compliant = ai_analysis.get("adgm_compliant", False)
                    compliance_status = "✅ Compliant" if compliant else "❌ Issues Found"
                    st.metric("📊 ADGM Compliance", compliance_status)
                    
                    st.metric("🔍 Issues Found", len(ai_analysis.get("compliance_issues", [])))
                
                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🔍 Issues", "💡 Suggestions", "📋 Key Findings", "🔧 Raw Analysis"])
                
                with tab1:
                    st.markdown("### 🚨 Compliance Issues")
                    issues = ai_analysis.get("compliance_issues", [])
                    if issues:
                        for i, issue in enumerate(issues, 1):
                            st.markdown(f"**{i}.** {issue}")
                    else:
                        st.success("✅ No major compliance issues detected!")
                
                with tab2:
                    st.markdown("### 💡 Recommendations")
                    suggestions = ai_analysis.get("suggestions", [])
                    if suggestions:
                        for i, suggestion in enumerate(suggestions, 1):
                            st.markdown(f"**{i}.** {suggestion}")
                    else:
                        st.info("No specific suggestions at this time.")
                
                with tab3:
                    st.markdown("### 📋 Key Findings")
                    findings = ai_analysis.get("key_findings", [])
                    if findings:
                        for i, finding in enumerate(findings, 1):
                            st.markdown(f"**{i}.** {finding}")
                    else:
                        st.info("No key findings reported.")
                
                with tab4:
                    st.markdown("### 🔧 Complete AI Analysis")
                    st.json(ai_analysis)
                
                # Download analysis report
                report_data = {
                    "filename": file.name,
                    "file_size": len(file_content),
                    "word_count": len(text_content.split()) if text_content else 0,
                    "ai_analysis": ai_analysis,
                    "analysis_timestamp": str(st.session_state.get('analysis_time', 'now'))
                }
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"{file.name}_analysis.json",
                    mime="application/json"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>ADGM Corporate Agent</strong> | Enhanced AI Analysis Version</p>
    <p><em>Powered by OpenAI GPT-3.5 for real document analysis</em></p>
</div>
""", unsafe_allow_html=True)