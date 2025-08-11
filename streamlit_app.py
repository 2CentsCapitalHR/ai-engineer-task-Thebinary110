# ADGM Corporate Agent - Fixed Streamlit Application
# Works WITHOUT LangChain - No more import errors!

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title='ADGM Corporate Agent - Fixed',
    page_icon='⚖️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for professional styling
st.markdown('''
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .compliance-high {
        border-left-color: #dc2626 !important;
        background: #fef2f2;
    }
    
    .compliance-medium {
        border-left-color: #f59e0b !important;
        background: #fffbeb;
    }
    
    .compliance-low {
        border-left-color: #10b981 !important;
        background: #f0fdf4;
    }
    
    .recommendation-item {
        background: #fefefe;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 3px solid #6366f1;
        color: #000000 !important;
    }
    
    .recommendation-item h4 {
        color: #000000 !important;
    }
    
    .status-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #000000 !important;
    }
    
    .status-card h3 {
        color: #000000 !important;
    }
    
    .status-card p {
        color: #000000 !important;
    }
</style>
''', unsafe_allow_html=True)

class SimpleCorporateAgent:
    """
    Simple Corporate Agent that works without LangChain
    Provides basic ADGM compliance checking
    """
    
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.has_google_ai = False
        
        # Try to import Google AI (optional)
        try:
            import google.generativeai as genai
            if self.google_api_key and self.google_api_key != 'your_google_api_key_here':
                genai.configure(api_key=self.google_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                self.has_google_ai = True
            else:
                self.model = None
        except ImportError:
            self.model = None
    
    def process_document(self, file_path: str, process_type: Optional[str] = None) -> Dict[str, Any]:
        """Process document without LangChain dependencies"""
        
        try:
            # Extract text from document
            doc_content = self._extract_text_from_docx(file_path)
            
            # Analyze document
            doc_analysis = self._analyze_document_type(doc_content, process_type)
            
            # Check compliance
            compliance_results = self._check_compliance(doc_content, doc_analysis)
            
            # Detect red flags
            red_flags = self._detect_red_flags(doc_content)
            
            # Check completeness
            completeness_check = self._check_completeness(doc_analysis['document_type'], doc_analysis['process_type'])
            
            # Generate report
            report = self._generate_report(doc_analysis, compliance_results, red_flags, completeness_check)
            
            return {
                'success': True,
                'document_analysis': doc_analysis,
                'compliance_results': compliance_results,
                'red_flags': red_flags,
                'completeness_check': completeness_check,
                'structured_report': report,
                'processing_timestamp': datetime.now().isoformat(),
                'agent_type': 'Simple (No LangChain)'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _extract_text_from_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = '\n'.join(paragraphs)
            
            return {
                'text': full_text,
                'paragraphs': paragraphs,
                'word_count': len(full_text.split())
            }
        except ImportError:
            raise Exception("python-docx not installed. Run: pip install python-docx")
        except Exception as e:
            raise Exception(f"Error reading document: {str(e)}")
    
    def _analyze_document_type(self, doc_content: Dict[str, Any], process_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """Simple document type analysis"""
        
        text = doc_content['text'].lower()
        
        # Pattern-based detection
        if 'articles of association' in text:
            doc_type = 'Articles of Association'
            process_type = 'company_incorporation'
        elif 'employment contract' in text or 'employment agreement' in text:
            doc_type = 'Employment Contract'
            process_type = 'employment'
        elif 'board resolution' in text:
            doc_type = 'Board Resolution'
            process_type = 'company_incorporation'
        elif 'memorandum of association' in text:
            doc_type = 'Memorandum of Association'
            process_type = 'company_incorporation'
        else:
            doc_type = 'Unknown Document'
            process_type = process_type_hint or 'general'
        
        # Detect jurisdiction
        if 'adgm' in text or 'abu dhabi global market' in text:
            jurisdiction = 'ADGM'
        elif 'dubai courts' in text or 'uae federal' in text:
            jurisdiction = 'UAE Federal/Dubai'
        else:
            jurisdiction = 'Not detected'
        
        return {
            'document_type': doc_type,
            'process_type': process_type,
            'confidence': 0.8 if doc_type != 'Unknown Document' else 0.3,
            'jurisdiction_mentioned': jurisdiction,
            'key_sections': [],
            'analysis_method': 'Pattern-based'
        }
    
    def _check_compliance(self, doc_content: Dict[str, Any], doc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance using rules"""
        
        text = doc_content['text'].lower()
        issues = []
        
        # Rule-based compliance checks
        if 'dubai courts' in text:
            issues.append({
                'issue': 'References Dubai Courts instead of ADGM Courts',
                'severity': 'High',
                'regulation': 'ADGM jurisdiction requirements',
                'recommendation': 'Update to reference ADGM Courts',
                'section': 'Jurisdiction'
            })
        
        if 'uae federal courts' in text:
            issues.append({
                'issue': 'References UAE Federal Courts instead of ADGM Courts',
                'severity': 'High',
                'regulation': 'ADGM jurisdiction requirements',
                'recommendation': 'Update to reference ADGM Courts',
                'section': 'Jurisdiction'
            })
        
        if 'adgm' not in text and 'abu dhabi global market' not in text:
            issues.append({
                'issue': 'No ADGM jurisdiction reference found',
                'severity': 'Medium',
                'regulation': 'ADGM legal framework',
                'recommendation': 'Add explicit ADGM jurisdiction clause',
                'section': 'Governing Law'
            })
        
        if 'signature' not in text and 'signed' not in text:
            issues.append({
                'issue': 'No signature section detected',
                'severity': 'High',
                'regulation': 'Document execution requirements',
                'recommendation': 'Add proper signature section',
                'section': 'Execution'
            })
        
        # Check for weak language
        weak_terms = ['may', 'might', 'possibly', 'perhaps', 'could']
        found_weak = [term for term in weak_terms if f' {term} ' in text]
        if found_weak:
            issues.append({
                'issue': f'Weak language found: {", ".join(found_weak)}',
                'severity': 'Medium',
                'regulation': 'Legal language standards',
                'recommendation': 'Use definitive language (shall, must, will)',
                'section': 'Language'
            })
        
        # Calculate score
        high_count = sum(1 for issue in issues if issue['severity'] == 'High')
        medium_count = sum(1 for issue in issues if issue['severity'] == 'Medium')
        
        score = max(0, 100 - (high_count * 25) - (medium_count * 10))
        
        return {
            'success': True,
            'overall_compliance_score': score,
            'issue_breakdown': {
                'high_priority': high_count,
                'medium_priority': medium_count,
                'low_priority': 0
            },
            'detailed_issues': issues,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _detect_red_flags(self, doc_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect red flags in document"""
        
        text = doc_content['text'].lower()
        red_flags = []
        
        # Jurisdiction red flags
        if 'dubai courts' in text:
            red_flags.append({
                'type': 'jurisdiction_error',
                'severity': 'High',
                'issue': 'References Dubai Courts instead of ADGM Courts',
                'suggestion': 'Update jurisdiction to ADGM Courts',
                'section': 'Jurisdiction'
            })
        
        if 'uae federal courts' in text:
            red_flags.append({
                'type': 'jurisdiction_error',
                'severity': 'High',
                'issue': 'References UAE Federal Courts instead of ADGM Courts',
                'suggestion': 'Update jurisdiction to ADGM Courts',
                'section': 'Jurisdiction'
            })
        
        # Missing ADGM reference
        if 'adgm' not in text and 'abu dhabi global market' not in text:
            red_flags.append({
                'type': 'missing_adgm_reference',
                'severity': 'Medium',
                'issue': 'No ADGM jurisdiction reference found',
                'suggestion': 'Add explicit ADGM jurisdiction clause',
                'section': 'Governing Law'
            })
        
        # Missing signatures
        if 'signature' not in text and 'signed' not in text:
            red_flags.append({
                'type': 'missing_signatures',
                'severity': 'High',
                'issue': 'No signature section detected',
                'suggestion': 'Add proper signature and execution section',
                'section': 'Document End'
            })
        
        # Weak language
        weak_terms = ['may', 'might', 'possibly', 'perhaps', 'could']
        found_weak = [term for term in weak_terms if f' {term} ' in text]
        if found_weak:
            red_flags.append({
                'type': 'ambiguous_language',
                'severity': 'Medium',
                'issue': f'Ambiguous language found: {", ".join(found_weak)}',
                'suggestion': 'Replace with definitive terms (shall, must, will)',
                'section': 'Language'
            })
        
        return red_flags
    
    def _check_completeness(self, document_type: str, process_type: str) -> Dict[str, Any]:
        """Check document completeness"""
        
        required_docs = {
            'company_incorporation': [
                'Articles of Association',
                'Memorandum of Association',
                'Board Resolution',
                'Incorporation Application Form'
            ],
            'employment': [
                'Employment Contract',
                'Job Description'
            ],
            'general': [
                'Main Document'
            ]
        }
        
        process_requirements = required_docs.get(process_type, required_docs['general'])
        uploaded_docs = [document_type] if document_type != 'Unknown Document' else []
        missing_docs = [doc for doc in process_requirements if doc not in uploaded_docs]
        
        return {
            'process_type': process_type,
            'required_documents': process_requirements,
            'uploaded_documents': uploaded_docs,
            'missing_documents': missing_docs,
            'completeness_percentage': (
                (len(uploaded_docs) / len(process_requirements)) * 100 
                if process_requirements else 100
            ),
            'is_complete': len(missing_docs) == 0
        }
    
    def _generate_report(self, doc_analysis: Dict[str, Any], compliance_results: Dict[str, Any], 
                        red_flags: List[Dict[str, Any]], completeness_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured report"""
        
        # Calculate risk score
        high_count = sum(1 for flag in red_flags if flag.get('severity') == 'High')
        medium_count = sum(1 for flag in red_flags if flag.get('severity') == 'Medium')
        risk_score = min(100, (high_count * 30) + (medium_count * 15))
        
        # Generate recommendations
        recommendations = []
        if any(flag['type'] == 'jurisdiction_error' for flag in red_flags):
            recommendations.append('🏛️ URGENT: Update jurisdiction clauses to reference ADGM Courts')
        
        if any(flag['type'] == 'missing_adgm_reference' for flag in red_flags):
            recommendations.append('📋 Add explicit ADGM governing law clause')
        
        if any(flag['type'] == 'ambiguous_language' for flag in red_flags):
            recommendations.append('✏️ Replace ambiguous language with definitive terms')
        
        if any(flag['type'] == 'missing_signatures' for flag in red_flags):
            recommendations.append('✍️ Add proper signature and execution section')
        
        if not completeness_check['is_complete']:
            recommendations.append(f'📁 Upload {len(completeness_check["missing_documents"])} missing documents')
        
        # Next steps
        next_steps = []
        if high_count > 0:
            next_steps.append('🚨 PRIORITY: Address all high-severity issues immediately')
        
        next_steps.extend([
            '📝 Review all flagged sections carefully',
            '⚖️ Consider legal review before submission',
            '📋 Complete any missing required documents'
        ])
        
        return {
            'document_info': {
                'document_type': doc_analysis.get('document_type', 'Unknown'),
                'process_type': doc_analysis.get('process_type', 'general'),
                'confidence': doc_analysis.get('confidence', 0.5),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'compliance_summary': {
                'total_issues': len(red_flags),
                'high_priority_issues': high_count,
                'medium_priority_issues': medium_count,
                'risk_score': risk_score,
                'overall_status': 'NEEDS_REVIEW' if risk_score > 30 else 'ACCEPTABLE'
            },
            'detailed_issues': red_flags,
            'completeness_analysis': completeness_check,
            'recommendations': recommendations,
            'next_steps': next_steps,
            'agent_info': {
                'type': 'Simple Agent (No LangChain)',
                'analysis_method': 'Rule-based compliance checking'
            }
        }

class ADGMCorporateAgentApp:
    """Fixed Streamlit application that works without LangChain"""
    
    def __init__(self):
        self.agent = None
        self._initialize_session_state()
        self._initialize_agent()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None
    
    def _initialize_agent(self):
        """Initialize the Simple Corporate Agent"""
        if 'agent' not in st.session_state:
            try:
                with st.spinner('🔧 Initializing ADGM Corporate Agent (Fixed Version)...'):
                    st.session_state.agent = SimpleCorporateAgent()
                    self.agent = st.session_state.agent
                    st.session_state.agent_status = '✅ Ready (No LangChain)'
            except Exception as e:
                st.session_state.agent_status = f'❌ Error: {str(e)}'
                self.agent = None
        else:
            self.agent = st.session_state.agent
    
    def run(self):
        """Main application runner"""
        self._render_header()
        self._render_sidebar()
        
        # Main content area
        if not st.session_state.processing_complete:
            self._render_upload_section()
        else:
            self._render_results_dashboard()
    
    def _render_header(self):
        """Render the main application header"""
        st.markdown('''
        <div class="main-header">
            <h1>⚖️ ADGM Corporate Agent - Fixed Version</h1>
            <p>Legal Document Intelligence for ADGM Compliance (No LangChain Dependencies)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with application info"""
        with st.sidebar:
            st.markdown('## 🏛️ About ADGM Agent')
            st.markdown('''
            **Fixed Features:**
            - 📄 Document Type Recognition
            - ⚖️ ADGM Compliance Checking  
            - 🚩 Red Flag Detection
            - 📊 Compliance Scoring
            - 💡 Recommendations
            - ✅ No LangChain Dependencies
            ''')
            
            st.markdown('---')
            
            # System status
            st.markdown('## 📈 System Status')
            agent_status = getattr(st.session_state, 'agent_status', '🔴 Agent Offline')
            if '✅' in agent_status:
                st.success(agent_status)
            else:
                st.error(agent_status)
            
            st.markdown('---')
            
            # Document categories
            st.markdown('## 📋 Supported Documents')
            with st.expander('Company Formation'):
                st.markdown('''
                - Articles of Association
                - Memorandum of Association
                - Board Resolutions
                - Incorporation Forms
                ''')
            
            with st.expander('Employment & HR'):
                st.markdown('''
                - Employment Contracts
                - HR Policies
                - Job Descriptions
                ''')
            
            # Reset button
            if st.session_state.processing_complete:
                if st.button('🔄 Process New Document', type='primary'):
                    self._reset_session()
    
    def _render_upload_section(self):
        """Render document upload section"""
        st.markdown('## 📤 Upload Document for Analysis')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                'Choose a .docx document',
                type=['docx'],
                help='Upload your legal document for ADGM compliance analysis'
            )
        
        with col2:
            process_type = st.selectbox(
                'Legal Process Type',
                [
                    'Auto-detect',
                    'Company Incorporation', 
                    'Employment Contract',
                    'Board Resolution',
                    'Licensing Application'
                ],
                help='Select the type of legal process'
            )
        
        if uploaded_file is not None:
            st.success(f'📄 File uploaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)')
            
            if not self.agent:
                st.error('❌ Corporate Agent not available.')
                return
            
            if st.button('🚀 Analyze Document', type='primary', use_container_width=True):
                self._process_document(uploaded_file, process_type)
    
    def _process_document(self, uploaded_file, process_type: str):
        """Process the uploaded document"""
        
        if not self.agent:
            st.error('❌ Corporate Agent not available.')
            return
        
        # Save uploaded file
        upload_dir = Path('temp/uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Processing with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            progress_bar.progress(20)
            status_text.text('🔄 Starting analysis...')
            time.sleep(0.5)
            
            progress_bar.progress(50)
            status_text.text('📄 Processing document...')
            time.sleep(0.5)
            
            progress_bar.progress(80)
            status_text.text('⚖️ Checking compliance...')
            
            # Process document
            result = self.agent.process_document(
                str(file_path), 
                process_type if process_type != 'Auto-detect' else None
            )
            
            progress_bar.progress(100)
            status_text.text('✅ Analysis complete!')
            
            if not result.get('success', False):
                st.error(f'❌ Processing failed: {result.get("error", "Unknown error")}')
                return
            
            # Store results
            st.session_state.analysis_results = result
            st.session_state.processing_complete = True
            
            # Clean up
            if file_path.exists():
                os.remove(file_path)
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f'❌ Processing failed: {str(e)}')
            if file_path.exists():
                os.remove(file_path)
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _render_results_dashboard(self):
        """Render results dashboard"""
        if not st.session_state.analysis_results:
            st.error('❌ No analysis results found')
            return
        
        result = st.session_state.analysis_results
        
        st.markdown(f'## 📊 Analysis Results: **{st.session_state.uploaded_file_name}**')
        
        # Quick status overview
        self._render_status_overview(result)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(['📋 Summary', '⚖️ Compliance', '🚩 Issues', '📊 Reports'])
        
        with tab1:
            self._render_summary_tab(result)
        
        with tab2:
            self._render_compliance_tab(result)
        
        with tab3:
            self._render_issues_tab(result)
        
        with tab4:
            self._render_reports_tab(result)
    
    def _render_status_overview(self, result: Dict[str, Any]):
        """Render status overview"""
        compliance_summary = result.get('structured_report', {}).get('compliance_summary', {})
        risk_score = compliance_summary.get('risk_score', 0)
        total_issues = compliance_summary.get('total_issues', 0)
        
        if risk_score <= 20:
            status_class = 'compliance-low'
            status_icon = '✅'
            status_text = 'COMPLIANT'
        elif risk_score <= 50:
            status_class = 'compliance-medium'
            status_icon = '⚠️'
            status_text = 'NEEDS REVIEW'
        else:
            status_class = 'compliance-high'
            status_icon = '🚨'
            status_text = 'MAJOR ISSUES'
        
        st.markdown(f'''
        <div class="status-card {status_class}">
            <h3>{status_icon} {status_text}</h3>
            <p><strong>Risk Score:</strong> {risk_score}/100 | <strong>Issues Found:</strong> {total_issues}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    def _render_summary_tab(self, result: Dict[str, Any]):
        """Render summary tab"""
        col1, col2, col3, col4 = st.columns(4)
        
        doc_analysis = result.get('document_analysis', {})
        compliance_results = result.get('compliance_results', {})
        completeness_check = result.get('completeness_check', {})
        
        with col1:
            st.metric('Document Type', doc_analysis.get('document_type', 'Unknown'))
        
        with col2:
            compliance_score = compliance_results.get('overall_compliance_score', 0)
            st.metric('Compliance Score', f'{compliance_score}/100')
        
        with col3:
            completeness_pct = completeness_check.get('completeness_percentage', 0)
            st.metric('Completeness', f'{completeness_pct:.1f}%')
        
        with col4:
            confidence = doc_analysis.get('confidence', 0)
            st.metric('AI Confidence', f'{confidence:.1%}')
        
        # Document info
        st.markdown('### 📄 Document Information')
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('**Document Information:**')
            st.write(f'• **Type:** {doc_analysis.get("document_type", "Unknown")}')
            st.write(f'• **Process:** {doc_analysis.get("process_type", "Not specified")}')
            st.write(f'• **Jurisdiction:** {doc_analysis.get("jurisdiction_mentioned", "Not detected")}')
        
        with col2:
            st.markdown('**Completeness Check:**')
            required_docs = completeness_check.get('required_documents', [])
            missing_docs = completeness_check.get('missing_documents', [])
            
            st.write(f'• **Required Documents:** {len(required_docs)}')
            st.write(f'• **Missing Documents:** {len(missing_docs)}')
            
            if missing_docs:
                st.warning('**Missing:**')
                for doc in missing_docs:
                    st.write(f'  - {doc}')
    
    def _render_compliance_tab(self, result: Dict[str, Any]):
        """Render compliance tab"""
        compliance_results = result.get('compliance_results', {})
        
        if not compliance_results:
            st.warning('⚠️ No compliance results available')
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score = compliance_results.get('overall_compliance_score', 0)
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = 'gauge+number',
                value = score,
                title = {'text': 'Compliance Score'},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': 'darkblue'},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgray'},
                        {'range': [50, 80], 'color': 'yellow'},
                        {'range': [80, 100], 'color': 'lightgreen'}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            issue_breakdown = compliance_results.get('issue_breakdown', {})
            
            if any(issue_breakdown.values()):
                issues_df = pd.DataFrame([
                    {'Severity': 'High Priority', 'Count': issue_breakdown.get('high_priority', 0)},
                    {'Severity': 'Medium Priority', 'Count': issue_breakdown.get('medium_priority', 0)},
                    {'Severity': 'Low Priority', 'Count': issue_breakdown.get('low_priority', 0)}
                ])
                
                fig = px.bar(issues_df, x='Severity', y='Count', title='Issues by Severity')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success('✅ No compliance issues detected!')
    
    def _render_issues_tab(self, result: Dict[str, Any]):
        """Render issues tab"""
        red_flags = result.get('red_flags', [])
        
        if not red_flags:
            st.success('🎉 No issues detected! Your document appears to be compliant.')
            return
        
        st.markdown('### 🚩 Identified Issues')
        
        for i, issue in enumerate(red_flags, 1):
            severity = issue.get('severity', 'Medium')
            icon = '🚨' if severity == 'High' else '⚠️' if severity == 'Medium' else 'ℹ️'
            
            with st.container():
                st.markdown(f'**{i}. {icon} {issue.get("issue", "Unknown issue")}**')
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f'**Section:** {issue.get("section", "Not specified")}')
                    st.write(f'**Type:** {issue.get("type", "general").replace("_", " ").title()}')
                
                with col2:
                    st.write(f'**Suggestion:** {issue.get("suggestion", "Please review")}')
                
                with col3:
                    if severity == 'High':
                        st.error(f'**{severity}**')
                    elif severity == 'Medium':
                        st.warning(f'**{severity}**')
                    else:
                        st.info(f'**{severity}**')
                
                st.divider()
        
        # Recommendations
        recommendations = result.get('structured_report', {}).get('recommendations', [])
        if recommendations:
            st.markdown('### 💡 Recommendations')
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f'''
                <div class="recommendation-item">
                    <h4>{i}. {rec}</h4>
                </div>
                ''', unsafe_allow_html=True)
    
    def _render_reports_tab(self, result: Dict[str, Any]):
        """Render reports tab"""
        st.markdown('### 📊 Detailed Analytics')
        
        structured_report = result.get('structured_report', {})
        
        if not structured_report:
            st.warning('⚠️ No structured report available')
            return
        
        # JSON viewer
        st.markdown('#### 📋 Complete Analysis Report')
        with st.expander('View Raw JSON Report', expanded=False):
            st.json(structured_report)
        
        # Next steps
        next_steps = structured_report.get('next_steps', [])
        if next_steps:
            st.markdown('#### 🎯 Next Steps')
            for i, step in enumerate(next_steps, 1):
                st.write(f'{i}. {step}')
        
        # Processing metadata
        st.markdown('#### 🔍 Processing Metadata')
        
        col1, col2 = st.columns(2)
        
        with col1:
            document_info = structured_report.get('document_info', {})
            if document_info:
                st.write('**Document Information:**')
                st.json(document_info)
        
        with col2:
            compliance_summary = structured_report.get('compliance_summary', {})
            if compliance_summary:
                st.write('**Compliance Summary:**')
                st.json(compliance_summary)
    
    def _reset_session(self):
        """Reset session state for new document processing"""
        st.session_state.processing_complete = False
        st.session_state.analysis_results = None
        st.session_state.uploaded_file_name = None
        st.rerun()

def main():
    """Main application entry point"""
    try:
        app = ADGMCorporateAgentApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("This is the fixed version that works without LangChain dependencies")
        
        # Show simple fallback
        st.markdown("## 🆘 Emergency Mode")
        st.write("If the app fails, you can still use basic text analysis:")
        
        text_input = st.text_area("Paste document text here:", height=200)
        if text_input and st.button("Quick Analysis"):
            # Simple analysis
            issues = []
            if "dubai courts" in text_input.lower():
                issues.append("🚨 Uses Dubai Courts - should be ADGM Courts")
            if "adgm" not in text_input.lower():
                issues.append("⚠️ Missing ADGM jurisdiction reference")
            if "signature" not in text_input.lower():
                issues.append("🚨 No signature section")
            
            if issues:
                st.subheader("Issues Found:")
                for issue in issues:
                    if "🚨" in issue:
                        st.error(issue)
                    else:
                        st.warning(issue)
            else:
                st.success("✅ No major issues detected!")

if __name__ == '__main__':
    main()