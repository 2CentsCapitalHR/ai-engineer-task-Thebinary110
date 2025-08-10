"""
Multi-Agent Orchestrator
Coordinates all specialized agents for document processing.
"""

import asyncio
from typing import Dict, List, Any, Optional
import time

from adgm_corporate_agent.agents.document_classifier import DocumentClassifierAgent
from adgm_corporate_agent.agents.compliance_checker import ComplianceCheckerAgent
from adgm_corporate_agent.agents.risk_assessor import RiskAssessorAgent
from adgm_corporate_agent.agents.suggestion_engine import SuggestionEngineAgent

from adgm_corporate_agent.utils.logger import setup_logger
from adgm_corporate_agent.utils.rag_engine import AdvancedRAGEngine
from adgm_corporate_agent.utils.document_processor import get_document_processor

logger = setup_logger(__name__)

class MultiAgentOrchestrator:
    """
    Orchestrates multiple AI agents for comprehensive document analysis.
    Coordinates classification, compliance checking, risk assessment, and suggestions.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the multi-agent orchestrator."""
        
        self.openai_api_key = openai_api_key
        self.model = model
        
        # Initialize agents
        self.classifier = DocumentClassifierAgent(openai_api_key, model)
        self.compliance_checker = ComplianceCheckerAgent(openai_api_key, model)
        self.risk_assessor = RiskAssessorAgent(openai_api_key, model)
        self.suggestion_engine = SuggestionEngineAgent(openai_api_key, model)
        
        # Initialize utilities
        self.rag_engine = None
        self.document_processor = get_document_processor()
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "total_issues_found": 0
        }
        
        logger.info("Multi-agent orchestrator initialized")

    async def initialize(self):
        """Initialize RAG engine and all agents."""
        try:
            # Initialize RAG engine
            self.rag_engine = AdvancedRAGEngine(
                openai_api_key=self.openai_api_key,
                embedding_model="text-embedding-3-small"
            )
            await self.rag_engine.build_knowledge_base()
            
            # Initialize all agents with RAG engine
            await self.compliance_checker.initialize(self.rag_engine)
            await self.risk_assessor.initialize(self.rag_engine)
            await self.suggestion_engine.initialize(self.rag_engine)
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise

    async def process_document(self, 
                             document_content: bytes, 
                             filename: str,
                             parsed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process document through all agents in coordinated workflow.
        
        Args:
            document_content: Raw document bytes
            filename: Original filename
            parsed_data: Pre-parsed document data (optional)
            
        Returns:
            Comprehensive analysis results
        """
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting orchestrated processing for {filename}")
            
            # Step 1: Parse document if not provided
            if not parsed_data:
                parsed_data = await self.document_processor.parse_document(
                    document_content, filename
                )
            
            # Step 2: Classify document (runs first as others depend on it)
            classification_result = await self.classifier.classify_document(
                document_text=parsed_data["full_text"],
                filename=filename,
                structure_info=parsed_data.get("structure", {})
            )
            
            # Step 3: Run parallel agent processing
            compliance_task = self.compliance_checker.check_compliance(
                document_text=parsed_data["full_text"],
                document_type=classification_result.document_type,
                parsed_data=parsed_data
            )
            
            risk_task = self.risk_assessor.assess_risks(
                document_text=parsed_data["full_text"],
                document_type=classification_result.document_type,
                classification_confidence=classification_result.confidence
            )
            
            # Wait for parallel tasks
            compliance_results, risk_results = await asyncio.gather(
                compliance_task, risk_task
            )
            
            # Step 4: Generate suggestions based on findings
            all_issues = compliance_results + risk_results
            suggestions = await self.suggestion_engine.generate_suggestions(
                document_text=parsed_data["full_text"],
                document_type=classification_result.document_type,
                issues=all_issues
            )
            
            # Step 5: Create inline comments for document
            comments = self._prepare_comments(all_issues, suggestions)
            modified_document = await self.document_processor.add_inline_comments(
                file_content=document_content,
                comments=comments,
                filename=filename
            )
            
            # Step 6: Compile final results
            processing_time = time.time() - start_time
            
            result = {
                "filename": filename,
                "success": True,
                "processing_time": processing_time,
                "classification": {
                    "document_type": classification_result.document_type,
                    "confidence": classification_result.confidence,
                    "supporting_evidence": classification_result.supporting_evidence,
                    "alternative_types": classification_result.alternative_types,
                    "process_category": classification_result.process_category
                },
                "compliance_issues": all_issues,
                "suggestions": suggestions,
                "modified_document": modified_document,
                "metadata": {
                    "word_count": parsed_data.get("word_count", 0),
                    "paragraph_count": parsed_data.get("paragraph_count", 0),
                    "tables_count": len(parsed_data.get("tables", [])),
                    "structure_analysis": parsed_data.get("structure", {})
                }
            }
            
            # Update statistics
            self._update_stats(processing_time, len(all_issues))
            
            logger.info(f"Completed processing {filename} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed for {filename}: {str(e)}")
            
            return {
                "filename": filename,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }

    def _prepare_comments(self, issues: List[Dict[str, Any]], 
                         suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare comments for inline insertion."""
        
        comments = []
        
        # Convert issues to comments
        for issue in issues:
            comment = {
                "paragraph_index": issue.get("paragraph_index", 0),
                "text": issue.get("description", ""),
                "severity": issue.get("severity", "Medium"),
                "suggestion": issue.get("suggestion", ""),
                "highlight_text": issue.get("highlight_text", ""),
                "adgm_reference": issue.get("adgm_reference", "")
            }
            
            # Add ADGM reference if available
            if comment["adgm_reference"]:
                comment["text"] += f" (Ref: {comment['adgm_reference']})"
            
            comments.append(comment)
        
        # Add general suggestions as comments
        for suggestion in suggestions:
            if suggestion.get("type") == "general":
                comment = {
                    "paragraph_index": suggestion.get("paragraph_index", 0),
                    "text": f"SUGGESTION: {suggestion.get('description', '')}",
                    "severity": "Low",
                    "suggestion": suggestion.get("implementation", ""),
                    "highlight_text": ""
                }
                comments.append(comment)
        
        return comments

    def _update_stats(self, processing_time: float, issues_found: int):
        """Update processing statistics."""
        
        self.stats["documents_processed"] += 1
        self.stats["total_processing_time"] += processing_time
        self.stats["total_issues_found"] += issues_found
        
        # Calculate average
        if self.stats["documents_processed"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / 
                self.stats["documents_processed"]
            )

    async def batch_process(self, 
                           documents: List[Dict[str, Any]],
                           max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Process multiple documents with concurrency control."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(doc_info):
            async with semaphore:
                return await self.process_document(
                    document_content=doc_info["content"],
                    filename=doc_info["filename"],
                    parsed_data=doc_info.get("parsed_data")
                )
        
        # Create tasks for all documents
        tasks = [process_single(doc) for doc in documents]
        
        # Execute with progress tracking
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            
            # Log progress
            completed = len(results)
            total = len(documents)
            logger.info(f"Batch progress: {completed}/{total} documents completed")
        
        return results

    async def validate_document_checklist(self, 
                                        document_types: List[str]) -> Dict[str, Any]:
        """Validate document checklist against ADGM requirements."""
        
        # Company incorporation checklist
        incorporation_checklist = [
            "Articles of Association",
            "Memorandum of Association",
            "Incorporation Application Form", 
            "UBO Declaration Form",
            "Register of Members and Directors"
        ]
        
        # Check for incorporation process
        incorporation_matches = [
            doc_type for doc_type in document_types 
            if doc_type in incorporation_checklist
        ]
        
        if len(incorporation_matches) >= 2:
            missing_docs = [
                doc for doc in incorporation_checklist 
                if doc not in document_types
            ]
            
            return {
                "process_detected": "Company Incorporation",
                "required_documents": incorporation_checklist,
                "present_documents": incorporation_matches,
                "missing_documents": missing_docs,
                "completion_percentage": len(incorporation_matches) / len(incorporation_checklist) * 100,
                "is_complete": len(missing_docs) == 0
            }
        
        return {
            "process_detected": "Unknown",
            "completion_percentage": 0,
            "is_complete": False
        }

    async def get_adgm_regulation_context(self, 
                                        query: str,
                                        document_type: str = None) -> List[Dict[str, Any]]:
        """Get relevant ADGM regulations for a specific query."""
        
        if not self.rag_engine:
            return []
        
        return await self.rag_engine.hierarchical_retrieve(
            query=query,
            document_type=document_type,
            max_results=5
        )

    async def rebuild_knowledge_base(self):
        """Rebuild the ADGM regulation knowledge base."""
        
        if self.rag_engine:
            await self.rag_engine.build_knowledge_base()
            logger.info("Knowledge base rebuilt successfully")

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        
        base_stats = self.stats.copy()
        
        # Add agent-specific stats
        base_stats.update({
            "classifier_stats": self.classifier.get_classification_stats(),
            "rag_collection_stats": self.rag_engine.get_collection_stats() if self.rag_engine else {},
            "cache_stats": self.document_processor.cache.get_stats() if hasattr(self.document_processor, 'cache') else {}
        })
        
        return base_stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        
        health = {
            "status": "healthy",
            "agents_initialized": True,
            "rag_engine_ready": self.rag_engine is not None,
            "openai_connection": True,  # Would test actual connection
            "issues": []
        }
        
        # Check each component
        try:
            if not self.rag_engine:
                health["issues"].append("RAG engine not initialized")
                health["status"] = "degraded"
            
            # Test classification
            test_result = await self.classifier.classify_document(
                "This is a test document", "test.docx"
            )
            if not test_result:
                health["issues"].append("Document classifier not responding")
                health["status"] = "degraded"
                
        except Exception as e:
            health["issues"].append(f"Health check failed: {str(e)}")
            health["status"] = "unhealthy"
        
        return health