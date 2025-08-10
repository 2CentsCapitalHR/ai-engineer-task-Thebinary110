"""
RAG Engine Implementation
Manages vector database and retrieval for ADGM regulations.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
import hashlib

from adgm_corporate_agent.utils.logger import setup_logger
from adgm_corporate_agent.utils.cache_manager import CacheManager

logger = setup_logger(__name__)

class AdvancedRAGEngine:
    """
    Advanced RAG engine with hierarchical retrieval and intelligent caching.
    Supports multi-level document processing and ADGM-specific optimizations.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "text-embedding-3-small"):
        """Initialize the RAG engine with advanced configurations."""
        
        self.openai_api_key = openai_api_key
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )
        
        # Initialize cache manager
        self.cache = CacheManager(cache_size=100)
        
        # Initialize ChromaDB
        self._setup_chromadb()
        
        # Initialize text splitter with legal-aware chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ".",     # Sentences
                ";",     # Clauses
                ",",     # Sub-clauses
                " "      # Words
            ]
        )
        
        # ADGM regulation URLs for scraping
        self.adgm_urls = [
            "https://www.adgm.com/registration-authority/registration-and-incorporation",
            "https://www.adgm.com/setting-up",
            "https://www.adgm.com/legal-framework/guidance-and-policy-statements",
            "https://www.adgm.com/operating-in-adgm/obligations-of-adgm-registered-entities"
        ]
        
        logger.info("RAG Engine initialized successfully")

    def _setup_chromadb(self):
        """Setup ChromaDB with persistent storage."""
        try:
            # Create ChromaDB client with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collections for different document types
            self.collections = {
                "adgm_regulations": self._get_or_create_collection("adgm_regulations"),
                "incorporation_docs": self._get_or_create_collection("incorporation_docs"),
                "compliance_rules": self._get_or_create_collection("compliance_rules"),
                "templates": self._get_or_create_collection("templates")
            }
            
            logger.info("ChromaDB setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {str(e)}")
            raise

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.chroma_client.get_collection(name=name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    async def scrape_adgm_content(self) -> List[Document]:
        """Scrape official ADGM content for RAG knowledge base."""
        documents = []
        
        for url in self.adgm_urls:
            try:
                logger.info(f"Scraping: {url}")
                
                # Check cache first
                cache_key = f"scraped_{hashlib.md5(url.encode()).hexdigest()}"
                cached_content = self.cache.get(cache_key)
                
                if cached_content:
                    documents.extend(cached_content)
                    continue
                
                # Scrape content
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract main content (customize selectors as needed)
                content_selectors = [
                    'main',
                    '.content',
                    '.main-content',
                    'article',
                    '[role="main"]'
                ]
                
                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = elements[0].get_text(strip=True)
                        break
                
                if not content:
                    content = soup.get_text(strip=True)
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "type": "adgm_regulation",
                        "scraped_at": str(asyncio.get_event_loop().time())
                    }
                )
                
                documents.append(doc)
                
                # Cache the scraped content
                self.cache.set(cache_key, [doc], ttl=86400)  # 24 hours
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                continue
        
        logger.info(f"Scraped {len(documents)} documents from ADGM sources")
        return documents

    def load_local_regulations(self, data_path: str = "adgm_corporate_agent/data/adgm_regulations") -> List[Document]:
        """Load local ADGM regulation files."""
        documents = []
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_path}")
            return documents
        
        for file_path in data_dir.glob("**/*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "type": "local_regulation",
                        "filename": file_path.name
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} local regulation documents")
        return documents

    def advanced_chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Advanced chunking with legal clause awareness."""
        chunked_docs = []
        
        for doc in documents:
            # Legal-aware splitting patterns
            legal_patterns = [
                r'\d+\.\d+\.',  # Section numbers (e.g., "3.1.")
                r'Article \d+',  # Article references
                r'Section \d+',  # Section references
                r'Clause \d+',   # Clause references
            ]
            
            # Split document
            chunks = self.text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                # Enhance metadata for better retrieval
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": self._classify_content_type(chunk.page_content)
                })
                
                chunked_docs.append(chunk)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def _classify_content_type(self, content: str) -> str:
        """Classify the type of legal content for better retrieval."""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['incorporation', 'company formation', 'articles of association']):
            return "incorporation"
        elif any(term in content_lower for term in ['compliance', 'regulation', 'requirement']):
            return "compliance"
        elif any(term in content_lower for term in ['template', 'form', 'application']):
            return "template"
        elif any(term in content_lower for term in ['procedure', 'process', 'step']):
            return "procedure"
        else:
            return "general"

    async def build_knowledge_base(self):
        """Build the complete RAG knowledge base."""
        logger.info("Building ADGM knowledge base...")
        
        # Collect all documents
        all_documents = []
        
        # Load local regulations
        local_docs = self.load_local_regulations()
        all_documents.extend(local_docs)
        
        # Scrape online content
        scraped_docs = await self.scrape_adgm_content()
        all_documents.extend(scraped_docs)
        
        if not all_documents:
            logger.warning("No documents found for knowledge base")
            return
        
        # Advanced chunking
        chunked_docs = self.advanced_chunk_documents(all_documents)
        
        # Generate embeddings and store in ChromaDB
        await self._store_documents(chunked_docs)
        
        logger.info("Knowledge base built successfully")

    async def _store_documents(self, documents: List[Document]):
        """Store documents in appropriate ChromaDB collections."""
        
        # Group documents by type
        doc_groups = {
            "adgm_regulations": [],
            "incorporation_docs": [],
            "compliance_rules": [],
            "templates": []
        }
        
        for doc in documents:
            content_type = doc.metadata.get("content_type", "general")
            doc_type = doc.metadata.get("type", "general")
            
            if content_type == "incorporation" or "incorporation" in doc.metadata.get("source", ""):
                doc_groups["incorporation_docs"].append(doc)
            elif content_type == "compliance" or content_type == "procedure":
                doc_groups["compliance_rules"].append(doc)
            elif content_type == "template":
                doc_groups["templates"].append(doc)
            else:
                doc_groups["adgm_regulations"].append(doc)
        
        # Store each group in its respective collection
        for collection_name, docs in doc_groups.items():
            if docs:
                await self._add_to_collection(collection_name, docs)

    async def _add_to_collection(self, collection_name: str, documents: List[Document]):
        """Add documents to a specific ChromaDB collection."""
        try:
            collection = self.collections[collection_name]
            
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"{collection_name}_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {str(e)}")

    async def hierarchical_retrieve(self, 
                                  query: str, 
                                  document_type: str = None,
                                  max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Hierarchical retrieval with document type awareness.
        Level 1: Document type filtering
        Level 2: Semantic similarity
        Level 3: Confidence scoring
        """
        
        # Level 1: Determine appropriate collections
        target_collections = self._select_collections(query, document_type)
        
        # Level 2: Perform semantic search across collections
        all_results = []
        
        for collection_name in target_collections:
            try:
                collection = self.collections[collection_name]
                
                # Generate query embedding
                query_embedding = await asyncio.to_thread(
                    self.embeddings.embed_query, query
                )
                
                # Search collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Process results
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "collection": collection_name,
                        "confidence": 1 - results["distances"][0][i]  # Convert distance to confidence
                    }
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Search failed for collection {collection_name}: {str(e)}")
        
        # Level 3: Sort by confidence and return top results
        all_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        logger.info(f"Retrieved {len(all_results[:max_results])} results for query: {query[:50]}...")
        
        return all_results[:max_results]

    def _select_collections(self, query: str, document_type: str = None) -> List[str]:
        """Select appropriate collections based on query and document type."""
        query_lower = query.lower()
        
        if document_type:
            if document_type.lower() in ["articles of association", "memorandum of association"]:
                return ["incorporation_docs", "adgm_regulations"]
            elif "resolution" in document_type.lower():
                return ["templates", "incorporation_docs"]
            elif "compliance" in document_type.lower():
                return ["compliance_rules", "adgm_regulations"]
        
        # Query-based selection
        if any(term in query_lower for term in ["incorporation", "company formation", "articles"]):
            return ["incorporation_docs", "adgm_regulations"]
        elif any(term in query_lower for term in ["compliance", "regulation", "rule"]):
            return ["compliance_rules", "adgm_regulations"]
        elif any(term in query_lower for term in ["template", "form"]):
            return ["templates", "adgm_regulations"]
        else:
            return list(self.collections.keys())  # Search all collections

    async def get_contextual_regulations(self, 
                                       document_content: str, 
                                       document_type: str) -> List[Dict[str, Any]]:
        """Get regulations contextual to specific document content."""
        
        # Extract key phrases from document
        key_phrases = self._extract_key_phrases(document_content)
        
        # Build contextual query
        contextual_query = f"{document_type} {' '.join(key_phrases[:5])}"
        
        # Retrieve relevant regulations
        results = await self.hierarchical_retrieve(
            query=contextual_query,
            document_type=document_type,
            max_results=10
        )
        
        return results

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key legal phrases from document text."""
        # Simple keyword extraction (can be enhanced with NLP)
        legal_keywords = [
            "jurisdiction", "directors", "shareholders", "registered address",
            "share capital", "articles", "memorandum", "resolution", 
            "incorporation", "compliance", "adgm", "liability"
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in legal_keywords if kw in text_lower]
        
        return found_keywords

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about stored documents."""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = count
            except:
                stats[name] = 0
        
        return stats

# Utility function for external use
async def initialize_rag_engine(openai_api_key: str) -> AdvancedRAGEngine:
    """Initialize and build the RAG engine."""
    engine = AdvancedRAGEngine(openai_api_key=openai_api_key)
    await engine.build_knowledge_base()
    return engine