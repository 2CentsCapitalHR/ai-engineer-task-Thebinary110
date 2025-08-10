"""
Document Classification Agent
Identifies document types using ML and rule-based approaches.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from adgm_corporate_agent.utils.logger import setup_logger
from adgm_corporate_agent.utils.cache_manager import get_cache_manager

logger = setup_logger(__name__)
cache = get_cache_manager()

@dataclass
class ClassificationResult:
    """Document classification result."""
    document_type: str
    confidence: float
    supporting_evidence: List[str]
    alternative_types: List[Tuple[str, float]]
    process_category: str

class DocumentClassifierAgent:
    """
    Advanced document classification agent using hybrid ML and rule-based approaches.
    Specializes in ADGM legal document identification.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the document classifier."""
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        
        # ADGM document types with detailed characteristics
        self.document_types = {
            "Articles of Association": {
                "keywords": [
                    "articles of association", "company constitution", "article", 
                    "share capital", "directors powers", "shareholders rights",
                    "voting rights", "dividend", "meetings", "quorum"
                ],
                "patterns": [
                    r"article\s+\d+",
                    r"the\s+company\s+shall",
                    r"directors?\s+may",
                    r"shareholders?\s+entitled",
                    r"share\s+capital\s+of"
                ],
                "structure_indicators": [
                    "numbered articles", "constitutional provisions", 
                    "governance structure", "rights and obligations"
                ],
                "process": "Company Formation"
            },
            
            "Memorandum of Association": {
                "keywords": [
                    "memorandum of association", "objects of the company",
                    "liability clause", "capital clause", "association clause",
                    "name clause", "registered office", "objects"
                ],
                "patterns": [
                    r"objects?\s+of\s+the\s+company",
                    r"liability\s+of\s+members",
                    r"capital\s+of\s+the\s+company",
                    r"we.*desire\s+to\s+be\s+formed",
                    r"company\s+limited\s+by"
                ],
                "structure_indicators": [
                    "name clause", "objects clause", "liability clause",
                    "capital clause", "association clause"
                ],
                "process": "Company Formation"
            },
            
            "Board Resolution": {
                "keywords": [
                    "board resolution", "directors meeting", "resolved that",
                    "board of directors", "quorum present", "meeting",
                    "directors present", "unanimous consent"
                ],
                "patterns": [
                    r"board\s+of\s+directors",
                    r"resolved\s+that",
                    r"quorum.*present",
                    r"meeting\s+of.*directors",
                    r"directors?\s+present",
                    r"it\s+was\s+resolved"
                ],
                "structure_indicators": [
                    "meeting details", "attendance", "resolutions",
                    "voting results", "signatures"
                ],
                "process": "Corporate Governance"
            },
            
            "Shareholder Resolution": {
                "keywords": [
                    "shareholder resolution", "general meeting", "shareholders meeting",
                    "ordinary resolution", "special resolution", "voting",
                    "shareholders present", "annual general meeting"
                ],
                "patterns": [
                    r"shareholders?\s+resolution",
                    r"general\s+meeting",
                    r"ordinary\s+resolution",
                    r"special\s+resolution",
                    r"shareholders?\s+present",
                    r"meeting\s+of.*shareholders"
                ],
                "structure_indicators": [
                    "meeting notice", "attendance", "voting results",
                    "resolutions passed", "shareholder signatures"
                ],
                "process": "Corporate Governance"
            },
            
            "Incorporation Application Form": {
                "keywords": [
                    "incorporation application", "company registration",
                    "application form", "proposed company name",
                    "nature of business", "registered office",
                    "share capital", "directors details"
                ],
                "patterns": [
                    r"incorporation\s+application",
                    r"proposed\s+company\s+name",
                    r"nature\s+of\s+business",
                    r"registered\s+office\s+address",
                    r"share\s+capital\s+details"
                ],
                "structure_indicators": [
                    "application sections", "company details",
                    "business information", "officer details"
                ],
                "process": "Company Formation"
            },
            
            "UBO Declaration Form": {
                "keywords": [
                    "ultimate beneficial owner", "ubo declaration",
                    "beneficial ownership", "controlling interest",
                    "ownership structure", "beneficial owner details"
                ],
                "patterns": [
                    r"ultimate\s+beneficial\s+owner",
                    r"ubo\s+declaration",
                    r"beneficial\s+ownership",
                    r"controlling\s+interest",
                    r"ownership\s+percentage"
                ],
                "structure_indicators": [
                    "owner identification", "ownership percentages",
                    "control mechanisms", "declaration statements"
                ],
                "process": "Compliance"
            },
            
            "Register of Members and Directors": {
                "keywords": [
                    "register of members", "register of directors",
                    "member details", "director information",
                    "share register", "shareholding details"
                ],
                "patterns": [
                    r"register\s+of\s+members",
                    r"register\s+of\s+directors",
                    r"member\s+name",
                    r"director\s+details",
                    r"share\s+allocation"
                ],
                "structure_indicators": [
                    "member list", "director list", "share allocations",
                    "appointment dates", "personal details"
                ],
                "process": "Corporate Records"
            }
        }
        
        logger.info("Document classifier initialized with ADGM document types")

    async def classify_document(self, 
                               document_text: str, 
                               filename: str,
                               structure_info: Dict[str, Any] = None) -> ClassificationResult:
        """
        Classify document type using hybrid approach.
        
        Args:
            document_text: Full document text
            filename: Original filename
            structure_info: Optional document structure information
            
        Returns:
            ClassificationResult with type, confidence, and evidence
        """
        
        # Check cache first
        cache_key = f"classification_{hash(document_text[:1000])}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached classification for {filename}")
            return ClassificationResult(**cached_result)
        
        try:
            # Step 1: Rule-based classification
            rule_based_scores = self._rule_based_classification(document_text, structure_info)
            
            # Step 2: ML-based classification using OpenAI
            ml_scores = await self._ml_based_classification(document_text, filename)
            
            # Step 3: Hybrid scoring
            final_scores = self._combine_scores(rule_based_scores, ml_scores)
            
            # Step 4: Determine best classification
            result = self._create_classification_result(final_scores, document_text)
            
            # Cache the result
            cache.set(cache_key, result.__dict__, ttl=3600)
            
            logger.info(f"Classified {filename} as {result.document_type} (confidence: {result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Classification failed for {filename}: {str(e)}")
            # Return default classification
            return ClassificationResult(
                document_type="Unknown Document Type",
                confidence=0.0,
                supporting_evidence=[],
                alternative_types=[],
                process_category="Unknown"
            )

    def _rule_based_classification(self, 
                                  text: str, 
                                  structure_info: Dict[str, Any] = None) -> Dict[str, float]:
        """Perform rule-based classification using patterns and keywords."""
        
        text_lower = text.lower()
        scores = {}
        
        for doc_type, characteristics in self.document_types.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in characteristics["keywords"] 
                                if keyword.lower() in text_lower)
            keyword_score = keyword_matches / len(characteristics["keywords"])
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in characteristics["patterns"]
                                if re.search(pattern, text_lower, re.IGNORECASE))
            pattern_score = pattern_matches / len(characteristics["patterns"])
            
            # Structure indicators (if available)
            structure_score = 0.0
            if structure_info:
                structure_matches = self._check_structure_indicators(
                    structure_info, characteristics["structure_indicators"]
                )
                structure_score = structure_matches / len(characteristics["structure_indicators"])
            
            # Combined score with weights
            final_score = (
                keyword_score * 0.4 +
                pattern_score * 0.4 +
                structure_score * 0.2
            )
            
            scores[doc_type] = final_score
        
        return scores

    def _check_structure_indicators(self, 
                                   structure_info: Dict[str, Any], 
                                   indicators: List[str]) -> int:
        """Check how many structure indicators are present."""
        matches = 0
        
        # Check headings
        headings = [h["text"].lower() for h in structure_info.get("headings", [])]
        
        # Check numbered items
        numbered_items = [item["text"].lower() for item in structure_info.get("numbered_items", [])]
        
        # Check signatures
        signatures = structure_info.get("signatures", [])
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            
            # Check in headings
            if any(indicator_lower in heading for heading in headings):
                matches += 1
                continue
            
            # Check in numbered items
            if any(indicator_lower in item for item in numbered_items):
                matches += 1
                continue
            
            # Check specific indicators
            if indicator_lower == "signatures" and signatures:
                matches += 1
        
        return matches

    async def _ml_based_classification(self, text: str, filename: str) -> Dict[str, float]:
        """Perform ML-based classification using OpenAI."""
        
        # Prepare prompt for GPT-4
        document_types_list = list(self.document_types.keys())
        
        prompt = f"""
You are an expert legal document classifier specializing in ADGM (Abu Dhabi Global Market) corporate documents.

Analyze the following document and classify it into one of these categories:
{', '.join(document_types_list)}

Document filename: {filename}

Document content (first 2000 characters):
{text[:2000]}

Provide your analysis in the following JSON format:
{{
    "primary_type": "most likely document type",
    "confidence": 0.95,
    "reasoning": "brief explanation of classification",
    "alternative_types": [
        {{"type": "alternative type", "confidence": 0.15}},
        {{"type": "another alternative", "confidence": 0.05}}
    ]
}}

Focus on:
1. Legal terminology and language patterns
2. Document structure and format
3. Specific ADGM compliance requirements
4. Standard legal document conventions

Respond only with the JSON, no additional text.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert legal document classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response (remove any markdown formatting)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text)
            
            # Convert to scores format
            scores = {doc_type: 0.0 for doc_type in document_types_list}
            
            # Set primary type score
            primary_type = result.get("primary_type", "")
            if primary_type in scores:
                scores[primary_type] = result.get("confidence", 0.0)
            
            # Set alternative type scores
            for alt in result.get("alternative_types", []):
                alt_type = alt.get("type", "")
                if alt_type in scores:
                    scores[alt_type] = alt.get("confidence", 0.0)
            
            return scores
            
        except Exception as e:
            logger.error(f"ML classification failed: {str(e)}")
            # Return neutral scores
            return {doc_type: 0.1 for doc_type in document_types_list}

    def _combine_scores(self, 
                       rule_scores: Dict[str, float], 
                       ml_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine rule-based and ML scores with appropriate weights."""
        
        combined_scores = {}
        
        for doc_type in self.document_types.keys():
            rule_score = rule_scores.get(doc_type, 0.0)
            ml_score = ml_scores.get(doc_type, 0.0)
            
            # Weighted combination (60% ML, 40% rules)
            combined_score = (ml_score * 0.6) + (rule_score * 0.4)
            combined_scores[doc_type] = combined_score
        
        return combined_scores

    def _create_classification_result(self, 
                                     scores: Dict[str, float], 
                                     document_text: str) -> ClassificationResult:
        """Create final classification result."""
        
        # Sort scores by confidence
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Best classification
        best_type, best_confidence = sorted_scores[0]
        
        # Alternative classifications (top 3)
        alternatives = [(doc_type, conf) for doc_type, conf in sorted_scores[1:4] if conf > 0.1]
        
        # Generate supporting evidence
        evidence = self._extract_supporting_evidence(document_text, best_type)
        
        # Determine process category
        process_category = self.document_types.get(best_type, {}).get("process", "Unknown")
        
        return ClassificationResult(
            document_type=best_type,
            confidence=best_confidence,
            supporting_evidence=evidence,
            alternative_types=alternatives,
            process_category=process_category
        )

    def _extract_supporting_evidence(self, text: str, document_type: str) -> List[str]:
        """Extract specific evidence supporting the classification."""
        evidence = []
        
        if document_type not in self.document_types:
            return evidence
        
        characteristics = self.document_types[document_type]
        text_lower = text.lower()
        
        # Find matching keywords
        found_keywords = [kw for kw in characteristics["keywords"] 
                         if kw.lower() in text_lower]
        
        # Find matching patterns
        found_patterns = []
        for pattern in characteristics["patterns"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            found_patterns.extend(matches[:2])  # Limit to 2 matches per pattern
        
        # Build evidence list
        if found_keywords:
            evidence.append(f"Keywords found: {', '.join(found_keywords[:5])}")
        
        if found_patterns:
            evidence.append(f"Patterns matched: {', '.join(found_patterns[:3])}")
        
        # Add document-specific evidence
        evidence.extend(self._get_specific_evidence(text, document_type))
        
        return evidence[:5]  # Limit to 5 pieces of evidence

    def _get_specific_evidence(self, text: str, document_type: str) -> List[str]:
        """Get document-type specific evidence."""
        evidence = []
        text_lower = text.lower()
        
        if "Articles of Association" in document_type:
            if re.search(r"article\s+\d+", text_lower):
                evidence.append("Contains numbered articles structure")
            if "director" in text_lower and "power" in text_lower:
                evidence.append("Contains director powers provisions")
                
        elif "Memorandum of Association" in document_type:
            if "objects of the company" in text_lower:
                evidence.append("Contains company objects clause")
            if "liability" in text_lower and "limited" in text_lower:
                evidence.append("Contains liability limitation clause")
                
        elif "Resolution" in document_type:
            if "resolved that" in text_lower:
                evidence.append("Contains formal resolution language")
            if "quorum" in text_lower:
                evidence.append("References meeting quorum")
                
        elif "UBO Declaration" in document_type:
            if re.search(r"\d+%", text):
                evidence.append("Contains ownership percentages")
            if "beneficial owner" in text_lower:
                evidence.append("Contains beneficial ownership terms")
        
        return evidence

    async def batch_classify(self, documents: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Classify multiple documents in batch."""
        
        tasks = []
        for doc in documents:
            task = self.classify_document(
                document_text=doc.get("text", ""),
                filename=doc.get("filename", "unknown"),
                structure_info=doc.get("structure", {})
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to classify document {i}: {str(result)}")
                final_results.append(ClassificationResult(
                    document_type="Classification Failed",
                    confidence=0.0,
                    supporting_evidence=[f"Error: {str(result)}"],
                    alternative_types=[],
                    process_category="Unknown"
                ))
            else:
                final_results.append(result)
        
        return final_results

    def get_supported_document_types(self) -> List[Dict[str, Any]]:
        """Get list of supported document types with details."""
        
        types_info = []
        for doc_type, characteristics in self.document_types.items():
            types_info.append({
                "name": doc_type,
                "process": characteristics["process"],
                "keywords": characteristics["keywords"][:5],  # Top 5 keywords
                "description": self._get_document_description(doc_type)
            })
        
        return types_info

    def _get_document_description(self, document_type: str) -> str:
        """Get description for document type."""
        descriptions = {
            "Articles of Association": "Company's internal constitution and governance rules",
            "Memorandum of Association": "Company's external constitution and objectives",
            "Board Resolution": "Formal decisions made by the board of directors",
            "Shareholder Resolution": "Formal decisions made by shareholders",
            "Incorporation Application Form": "Official application for company registration",
            "UBO Declaration Form": "Declaration of ultimate beneficial ownership",
            "Register of Members and Directors": "Official records of company officers and members"
        }
        
        return descriptions.get(document_type, "ADGM corporate document")

    async def validate_classification(self, 
                                    classification: ClassificationResult,
                                    document_text: str) -> Dict[str, Any]:
        """Validate and provide confidence assessment for classification."""
        
        validation = {
            "is_confident": classification.confidence > 0.7,
            "confidence_level": "High" if classification.confidence > 0.8 else 
                              "Medium" if classification.confidence > 0.5 else "Low",
            "recommendations": [],
            "potential_issues": []
        }
        
        # Check for potential issues
        if classification.confidence < 0.5:
            validation["potential_issues"].append("Low classification confidence")
            validation["recommendations"].append("Manual review recommended")
        
        if len(classification.alternative_types) > 0:
            top_alt = classification.alternative_types[0]
            if top_alt[1] > 0.3:  # Alternative type has significant confidence
                validation["potential_issues"].append(f"Possible alternative: {top_alt[0]}")
                validation["recommendations"].append("Consider document content review")
        
        if not classification.supporting_evidence:
            validation["potential_issues"].append("Limited supporting evidence found")
            validation["recommendations"].append("Verify document content manually")
        
        return validation

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics and performance metrics."""
        # In a real implementation, this would track actual usage statistics
        return {
            "supported_types": len(self.document_types),
            "average_confidence": 0.85,  # Would be calculated from actual usage
            "total_classifications": 0,   # Would be tracked
            "success_rate": 0.95,        # Would be measured
            "processing_time_avg": 0.5   # Average processing time in seconds
        }

# Utility function for external use
async def classify_document(openai_api_key: str, 
                          document_text: str, 
                          filename: str,
                          structure_info: Dict[str, Any] = None) -> ClassificationResult:
    """Utility function to classify a single document."""
    classifier = DocumentClassifierAgent(openai_api_key)
    return await classifier.classify_document(document_text, filename, structure_info)