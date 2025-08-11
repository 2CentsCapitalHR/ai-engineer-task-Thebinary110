import os
from pathlib import Path
from typing import List, Dict

class Config:
    # API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # Project Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))
    
    # Vector Store
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', './data/vector_store')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'adgm_legal_docs')
    
    # Document Processing
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10485760'))
    SUPPORTED_FORMATS = ['.docx', '.pdf']
    
    # ADGM Document Categories
    DOCUMENT_CATEGORIES = {
        'company_formation': [
            'Articles of Association',
            'Memorandum of Association',
            'Board Resolution Templates',
            'Shareholder Resolution Templates',
            'Incorporation Application Form',
            'UBO Declaration Form',
            'Register of Members and Directors'
        ],
        'employment': [
            'Standard Employment Contract',
            'Employment Terms',
            'HR Policies'
        ],
        'compliance': [
            'Annual Accounts',
            'Compliance Filings',
            'Data Protection Policies'
        ]
    }

# Default configuration
config = Config()
