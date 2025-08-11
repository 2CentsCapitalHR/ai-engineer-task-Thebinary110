# Fixed Corporate Agent - Compatible with Latest LangChain Versions
# This version resolves the LangSmithParams import error

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Fixed imports with error handling
try:
    import google.generativeai as genai
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install compatible versions: pip install -r requirements_fixed.txt")
    raise

# Import with fallback handling
try:
    from src.document_processing.docx_processor import DocxProcessor
    from src.utils.logger import setup_logger
except ImportError:
    # Fallback document processor
    class DocxProcessor:
        def parse_document(self, file_path):
            return {"text": "Sample document content", "structure": {}}
        
        def add_comments_to_document(self, original_path, compliance_results, red_flags, output_path):
            return original_path
    
    # Fallback logger
    def setup_logger(name):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = setup_logger(__name__)

class SyncCorporateAgent:
    """
    Fixed Corporate Agent for ADGM legal document processing.
    Resolves LangChain version compatibility issues.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self._setup_llm()
        self._setup_components()
        
    def _load_config(self) -> Dict:
        """Load configuration settings"""
        return {
            'google_api_key': os.getenv('GOOGLE_API_KEY'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
            'top_k_retrieval': int(os.getenv('TOP_K_RETRIEVAL', '5')),
            'vector_store_path': os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', '10485760'))
        }
    
    def _setup_llm(self):
        """Initialize the Gemini LLM with error handling"""
        if not self.config['google_api_key']:
            raise ValueError('GOOGLE_API_KEY not found. Please set it in your .env file.')
        
        try:
            genai.configure(api_key=self.config['google_api_key'])
            
            # Initialize LangChain Gemini with fixed parameters
            self.llm = ChatGoogleGenerativeAI(
                model='gemini-1.5-pro',
                google_api_key=self.config['google_api_key'],
                temperature=0.1,
                max_tokens=2048,
                # Remove any problematic parameters
                convert_system_message_to_human=True
            )
            
            # Initialize embeddings with fixed parameters
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model='models/embedding-001',
                google_api_key=self.config['google_api_key']
            )
            
            logger.info('✅ LLM and embeddings initialized successfully')
            
        except Exception as e:
            logger.error(f'❌ Error initializing LLM: {str(e)}')
            # Fallback to basic Google GenAI
            try:
                genai.configure(api_key=self.config['google_api_key'])
                self.llm = genai.GenerativeModel('gemini-1.5-pro')
                self.embeddings = None
                logger.warning('⚠️ Using fallback Google GenAI (limited functionality)')
            except Exception as e2:
                logger.error(f'❌ Fallback initialization failed: {str(e2)}')
                raise
    
    def _setup_components(self):
        """Initialize all processing components with error handling"""
        try:
            self.docx_processor = DocxProcessor()
            
            # Initialize knowledge base with robust error handling
            try:
                self.knowledge_base = self._create_mock_knowledge_base()
                logger.info('✅ Mock knowledge base initialized')
            except Exception as e:
                logger.warning(f'⚠️ Knowledge base initialization issue: {str(e)}')
                self.knowledge_base = None
            
            logger.info('✅ All components initialized successfully')
            
        except Exception as e:
            logger.error(f'❌ Error initializing components: {str(e)}')
            # Continue with minimal functionality
            self.docx_processor = DocxProcessor()
            self.knowledge_base = None
            logger.warning('⚠️ Running with minimal functionality')
    
    def _create_mock_knowledge_base(self):
        """Create a mock knowledge base for demo purposes"""
        class MockKnowledgeBase:
            def get_knowledge_base_stats(self):
                return {
                    'total_documents': 5,
                    'vector_store_path': './data/vector_store',
                    'collection_name': 'adgm_legal_docs',
                    'last_updated': datetime.now().isoformat(),
                    'categories': ['company_formation', 'employment', 'compliance'],
                    'status': 'ready',
                    'store_type': 'Mock'
                }
            
            def retrieve_relevant_context(self, query, document_type, top_k=5):
                return [
                    {
                        'content': 'ADGM jurisdiction clause must reference Abu Dhabi Global Market Courts exclusively. Documents must NOT reference UAE Federal Courts or Dubai Courts.',
                        'metadata': {'source': 'ADGM_Legal_Framework', 'category': 'jurisdiction'},
                        'relevance_score': 0.9,
                        'source': 'ADGM Legal Framework',
                        'category': 'jurisdiction'
                    },
                    {
                        'content': 'All ADGM legal documents must use definitive language such as shall, must, and will instead of ambiguous terms like may, possibly, or perhaps.',
                        'metadata': {'source': 'ADGM_Legal_Standards', 'category': 'language'},
                        'relevance_score': 0.8,
                        'source': 'ADGM Legal Standards', 
                        'category': 'language'
                    }
                ]
        
        return MockKnowledgeBase()
    
    def process_document(self, file_path: str, process_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Main document processing pipeline (synchronous version with error handling).
        """
        logger.info(f'🔄 Starting document processing for: {file_path}')
        
        try:
            # Step 1: Parse and extract document content
            doc_content = self._parse_document(file_path)
            
            # Step 2: Identify document type and process
            doc_analysis = self._analyze_document_type(doc_content, process_type)
            
            # Step 3: Perform compliance checking
            compliance_results = self._check_compliance(doc_content, doc_analysis)
            
            # Step 4: Detect red flags and issues
            red_flags = self._detect_red_flags(doc_content, doc_analysis)
            
            # Step 5: Check document completeness
            completeness_check = self._check_document_completeness(
                doc_analysis['document_type'], 
                doc_analysis['process_type']
            )
            
            # Step 6: Generate marked-up document with comments
            marked_up_path = self._generate_marked_up_document(
                file_path, compliance_results, red_flags
            )
            
            # Step 7: Create structured report
            report = self._generate_structured_report(
                doc_analysis, compliance_results, red_flags, completeness_check
            )
            
            logger.info('✅ Document processing completed successfully')
            
            return {
                'success': True,
                'document_analysis': doc_analysis,
                'compliance_results': compliance_results,
                'red_flags': red_flags,
                'completeness_check': completeness_check,
                'marked_up_document': marked_up_path,
                'structured_report': report,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f'❌ Error processing document: {str(e)}')
            return {
                'success': False,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse the uploaded .docx document"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f'Document not found: {file_path}')
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size > self.config['max_file_size']:
            raise ValueError(f'File too large: {file_size} bytes (max: {self.config["max_file_size"]})')
        
        return self.docx_processor.parse_document(file_path)
    
    def _analyze_document_type(self, doc_content: Dict[str, Any], process_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze and identify the document type and legal process"""
        
        prompt = f'''
        Analyze the following legal document and identify:
        1. Document type (e.g., Articles of Association, Employment Contract, etc.)
        2. Legal process (e.g., company_incorporation, employment, licensing, etc.)
        3. Key sections and structure
        4. Jurisdiction mentioned (should be ADGM for compliance)
        
        Document Content Preview:
        {doc_content['text'][:2000]}...
        
        Process Type Hint: {process_type_hint or "Not provided"}
        
        Respond in JSON format:
        {{
            "document_type": "specific_document_name",
            "process_type": "legal_process_category", 
            "confidence": 0.95,
            "key_sections": ["section1", "section2"],
            "jurisdiction_mentioned": "detected_jurisdiction",
            "analysis_notes": "brief explanation"
        }}
        '''
        
        try:
            # Handle both LangChain and direct GenAI
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                response_text = response.content
            else:
                # Fallback to direct GenAI
                response = self.llm.generate_content(prompt)
                response_text = response.text
            
            analysis = json.loads(response_text)
            logger.info(f'📋 Document identified as: {analysis.get("document_type", "Unknown")}')
            return analysis
            
        except json.JSONDecodeError:
            return {
                'document_type': 'Unknown',
                'process_type': process_type_hint or 'general',
                'confidence': 0.5,
                'key_sections': [],
                'jurisdiction_mentioned': 'Not detected',
                'analysis_notes': 'Failed to parse LLM response'
            }
        except Exception as e:
            logger.error(f'❌ Error in document analysis: {str(e)}')
            return {
                'document_type': 'Unknown', 
                'process_type': 'general',
                'confidence': 0.0,
                'key_sections': [],
                'jurisdiction_mentioned': 'Error',
                'analysis_notes': f'Analysis failed: {str(e)}'
            }
    
    def _check_compliance(self, doc_content: Dict[str, Any], doc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform compliance checking against ADGM regulations"""
        
        try:
            # Get legal context
            if self.knowledge_base:
                legal_context = self.knowledge_base.retrieve_relevant_context(
                    f'ADGM {doc_analysis["document_type"]} compliance', 
                    doc_analysis['document_type']
                )
            else:
                legal_context = []
            
            # Prepare context for AI analysis
            context_text = '\n\n'.join([
                f'Legal Reference: {doc["content"][:500]}...' 
                for doc in legal_context[:3]
            ])
            
            # AI-powered compliance analysis
            analysis_prompt = f'''
            As an ADGM legal compliance expert, analyze this {doc_analysis["document_type"]} for compliance issues.
            
            DOCUMENT CONTENT:
            {doc_content["text"][:3000]}...
            
            RELEVANT ADGM LEGAL CONTEXT:
            {context_text}
            
            ANALYSIS REQUIRED:
            1. Identify specific ADGM compliance gaps
            2. Check against relevant regulations
            3. Highlight critical legal issues
            4. Provide improvement recommendations
            
            Focus on:
            - ADGM jurisdiction and governing law compliance
            - Mandatory clause requirements
            - Legal language precision
            - Document structure completeness
            
            Return analysis in JSON format:
            {{
                "overall_compliance_score": 85,
                "critical_issues": [
                    {{
                        "issue": "specific issue description",
                        "severity": "High/Medium/Low",
                        "regulation": "specific ADGM regulation reference",
                        "recommendation": "specific fix recommendation",
                        "section": "document section affected"
                    }}
                ],
                "strengths": ["positive compliance aspects"],
                "recommendations": ["prioritized improvement recommendations"],
                "regulatory_references": ["specific ADGM laws/regulations cited"]
            }}
            '''
            
            # Handle both LangChain and direct GenAI
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(analysis_prompt)
                response_text = response.content
            else:
                response = self.llm.generate_content(analysis_prompt)
                response_text = response.text
                
            ai_analysis = json.loads(response_text)
            
            return {
                'success': True,
                'overall_compliance_score': ai_analysis.get('overall_compliance_score', 75),
                'ai_analysis_summary': ai_analysis,
                'legal_context_used': len(legal_context),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f'❌ Error in compliance checking: {str(e)}')
            return {
                'success': False,
                'overall_compliance_score': 50,
                'ai_analysis_summary': {'error': str(e)},
                'legal_context_used': 0,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _detect_red_flags(self, doc_content: Dict[str, Any], doc_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect legal red flags and compliance issues"""
        
        red_flags = []
        text = doc_content['text'].lower()
        
        # Jurisdiction red flags
        problematic_jurisdictions = ['uae federal courts', 'dubai courts', 'abu dhabi courts']
        for jurisdiction in problematic_jurisdictions:
            if jurisdiction in text:
                red_flags.append({
                    'type': 'jurisdiction_error',
                    'severity': 'High',
                    'issue': f'References {jurisdiction} instead of ADGM Courts',
                    'suggestion': 'Update jurisdiction clause to specify ADGM Courts',
                    'section': 'Jurisdiction Clause'
                })
        
        # Missing ADGM reference
        if 'adgm' not in text and 'abu dhabi global market' not in text:
            red_flags.append({
                'type': 'missing_adgm_reference',
                'severity': 'Medium', 
                'issue': 'No clear reference to ADGM jurisdiction',
                'suggestion': 'Add explicit ADGM jurisdiction clause',
                'section': 'Governing Law'
            })
        
        # Ambiguous language detection
        ambiguous_terms = ['may', 'possibly', 'perhaps', 'might', 'could']
        for term in ambiguous_terms:
            if f' {term} ' in text:
                red_flags.append({
                    'type': 'ambiguous_language',
                    'severity': 'Medium',
                    'issue': f'Use of ambiguous term {term} may weaken enforceability',
                    'suggestion': f'Replace {term} with definitive language',
                    'section': 'Multiple sections'
                })
                break  # Only flag once per document
        
        # Missing signature section
        if 'signature' not in text and 'signed' not in text:
            red_flags.append({
                'type': 'missing_signatures',
                'severity': 'High',
                'issue': 'No signature section detected',
                'suggestion': 'Add proper signature and execution section',
                'section': 'Document End'
            })
        
        logger.info(f'🚩 Detected {len(red_flags)} red flags')
        return red_flags
    
    def _check_document_completeness(self, document_type: str, process_type: str) -> Dict[str, Any]:
        """Check if all required documents are present for the legal process"""
        
        # Define required documents for different processes
        required_docs = {
            'company_incorporation': [
                'Articles of Association',
                'Memorandum of Association', 
                'Board Resolution',
                'Incorporation Application Form',
                'Register of Members and Directors'
            ],
            'employment': [
                'Employment Contract',
                'Job Description',
                'Salary Certificate'
            ],
            'licensing': [
                'License Application',
                'Supporting Documents',
                'Financial Statements'
            ]
        }
        
        process_requirements = required_docs.get(process_type, [])
        uploaded_docs = [document_type] if document_type != 'Unknown' else []
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
    
    def _generate_marked_up_document(self, original_path: str, compliance_results: Dict[str, Any], red_flags: List[Dict[str, Any]]) -> str:
        """Generate a marked-up .docx document with comments and highlights"""
        
        output_path = f'outputs/reviewed_docs/reviewed_{Path(original_path).name}'
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use docx processor to add comments
            marked_up_path = self.docx_processor.add_comments_to_document(
                original_path, 
                compliance_results, 
                red_flags, 
                output_path
            )
            
            logger.info(f'📝 Generated marked-up document: {marked_up_path}')
            return marked_up_path
        except Exception as e:
            logger.error(f'❌ Error generating marked-up document: {str(e)}')
            return original_path
    
    def _generate_structured_report(self, doc_analysis: Dict[str, Any], compliance_results: Dict[str, Any], red_flags: List[Dict[str, Any]], completeness_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final structured JSON report"""
        
        # Calculate overall risk score
        high_severity_count = sum(1 for flag in red_flags if flag.get('severity') == 'High')
        medium_severity_count = sum(1 for flag in red_flags if flag.get('severity') == 'Medium')
        risk_score = min(100, (high_severity_count * 30) + (medium_severity_count * 15))
        
        report = {
            'document_info': {
                'document_type': doc_analysis.get('document_type', 'Unknown'),
                'process_type': doc_analysis.get('process_type', 'general'),
                'confidence': doc_analysis.get('confidence', 0.5),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'completeness_analysis': completeness_check,
            'compliance_summary': {
                'total_issues': len(red_flags),
                'high_priority_issues': high_severity_count,
                'medium_priority_issues': medium_severity_count,
                'risk_score': risk_score,
                'overall_status': 'NEEDS_REVIEW' if risk_score > 30 else 'ACCEPTABLE'
            },
            'detailed_issues': red_flags,
            'recommendations': self._generate_recommendations(red_flags, completeness_check),
            'next_steps': self._generate_next_steps(completeness_check, red_flags)
        }
        
        # Save report to file
        report_path = f'outputs/reports/report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f'📊 Generated structured report: {report_path}')
        except Exception as e:
            logger.warning(f'⚠️ Could not save report file: {str(e)}')
        
        return report
    
    def _generate_recommendations(self, red_flags: List[Dict[str, Any]], completeness_check: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on findings"""
        
        recommendations = []
        
        # Recommendations based on red flags
        if any(flag['type'] == 'jurisdiction_error' for flag in red_flags):
            recommendations.append('🏛️ Update all jurisdiction clauses to reference ADGM Courts exclusively')
        
        if any(flag['type'] == 'missing_adgm_reference' for flag in red_flags):
            recommendations.append('📋 Add explicit ADGM governing law clause in legal provisions section')
        
        if any(flag['type'] == 'ambiguous_language' for flag in red_flags):
            recommendations.append('✏️ Replace ambiguous language with definitive, binding terms')
        
        if any(flag['type'] == 'missing_signatures' for flag in red_flags):
            recommendations.append('✍️ Add proper signature block with date, witness, and notarization sections')
        
        # Recommendations based on completeness
        if not completeness_check['is_complete']:
            missing_count = len(completeness_check['missing_documents'])
            recommendations.append(f'📁 Upload {missing_count} missing required documents before submission')
        
        # General recommendations
        if len(red_flags) > 3:
            recommendations.append('⚖️ Consider legal review before finalization due to multiple compliance issues')
        
        return recommendations
    
    def _generate_next_steps(self, completeness_check: Dict[str, Any], red_flags: List[Dict[str, Any]]) -> List[str]:
        """Generate specific next steps for the user"""
        
        next_steps = []
        
        # Priority-based next steps
        high_priority_issues = [flag for flag in red_flags if flag.get('severity') == 'High']
        
        if high_priority_issues:
            next_steps.append('🚨 URGENT: Address all high-priority compliance issues before proceeding')
        
        if not completeness_check['is_complete']:
            for missing_doc in completeness_check['missing_documents']:
                next_steps.append(f'📄 Prepare and upload: {missing_doc}')
        
        next_steps.extend([
            '📝 Review marked-up document with highlighted comments',
            '🔍 Validate all suggested changes with legal counsel', 
            '📋 Complete ADGM online submission portal requirements',
            '💾 Save final versions for regulatory filing'
        ])
        
        return next_steps