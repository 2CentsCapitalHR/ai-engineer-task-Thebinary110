"""
Compliance Validation Agent
Validates documents against ADGM regulations using RAG.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

from adgm_corporate_agent.utils.logger import setup_logger
from adgm_corporate_agent.utils.cache_manager import get_cache_manager

logger = setup_logger(__name__)
cache = get_cache_manager()

@dataclass
class ComplianceIssue:
    """Represents a compliance issue found in a document."""
    description: str
    severity: str  # Critical, High, Medium, Low
    section: str
    adgm_reference: str
    suggestion: str
    paragraph_index: int = 0
    highlight_text: str = ""

class ComplianceCheckerAgent:
    """
    Advanced compliance checking agent using RAG-powered ADGM regulation validation.
    Identifies regulatory violations and provides specific ADGM references.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the compliance checker."""
        
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.rag_engine = None
        
        # ADGM compliance rules organized by document type
        self.compliance_rules = {
            "Articles of Association": {
                "mandatory_clauses": [
                    "jurisdiction_clause",
                    "registered_office", 
                    "share_capital",
                    "directors_powers",
                    "shareholders_rights"
                ],
                "prohibited_terms": [
                    "uae federal courts",
                    "dubai courts", 
                    "abu dhabi courts"
                ],
                "required_patterns": [
                    r"adgm\s+courts?\s+shall\s+have\s+jurisdiction",
                    r"registered\s+office.*adgm",
                    r"share\s+capital"
                ]
            },
            
            "Memorandum of Association": {
                "mandatory_clauses": [
                    "name_clause",
                    "objects_clause",
                    "liability_clause", 
                    "capital_clause",
                    "association_clause"
                ],
                "required_patterns": [
                    r"objects?\s+of\s+the\s+company",
                    r"liability.*limited",
                    r"capital.*company"
                ]
            },
            
            "Board Resolution": {
                "mandatory_elements": [
                    "meeting_details",
                    "quorum_confirmation",
                    "resolution_text",
                    "voting_results",
                    "signatures"
                ],
                "required_patterns": [
                    r"quorum.*present",
                    r"resolved\s+that",
                    r"unanimously\s+approved|majority\s+approved"
                ]
            },
            
            "Shareholder Resolution": {
                "mandatory_elements": [
                    "meeting_notice",
                    "attendance_record", 
                    "resolution_details",
                    "voting_results",
                    "chairperson_signature"
                ]
            },
            
            "Incorporation Application Form": {
                "required_sections": [
                    "company_name",
                    "registered_office",
                    "business_nature",
                    "share_capital",
                    "directors_details",
                    "shareholders_details"
                ]
            },
            
            "UBO Declaration Form": {
                "required_information": [
                    "beneficial_owner_details",
                    "ownership_percentage", 
                    "control_mechanisms",
                    "declaration_statement"
                ],
                "percentage_validation": True
            }
        }
        
        logger.info("Compliance checker initialized")

    async def initialize(self, rag_engine):
        """Initialize with RAG engine for regulation retrieval."""
        self.rag_engine = rag_engine
        logger.info("Compliance checker connected to RAG engine")

    async def check_compliance(self, 
                              document_text: str,
                              document_type: str,
                              parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform comprehensive compliance check on document.
        
        Args:
            document_text: Full document text
            document_type: Classified document type
            parsed_data: Structured document data
            
        Returns:
            List of compliance issues found
        """
        
        # Check cache first
        cache_key = f"compliance_{hash(document_text[:1000])}_{document_type}"
        cached_issues = cache.get(cache_key)
        if cached_issues:
            logger.info("Using cached compliance results")
            return cached_issues
        
        try:
            logger.info(f"Starting compliance check for {document_type}")
            
            all_issues = []
            
            # Step 1: Rule-based compliance checks
            rule_issues = await self._rule_based_compliance_check(
                document_text, document_type, parsed_data
            )
            all_issues.extend(rule_issues)
            
            # Step 2: RAG-powered regulation validation
            if self.rag_engine:
                rag_issues = await self._rag_powered_compliance_check(
                    document_text, document_type
                )
                all_issues.extend(rag_issues)
            
            # Step 3: AI-powered deep analysis
            ai_issues = await self._ai_powered_compliance_analysis(
                document_text, document_type
            )
            all_issues.extend(ai_issues)
            
            # Step 4: Deduplicate and prioritize issues
            final_issues = self._deduplicate_issues(all_issues)
            
            # Cache results
            cache.set(cache_key, final_issues, ttl=1800)  # 30 minutes
            
            logger.info(f"Found {len(final_issues)} compliance issues")
            return final_issues
            
        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return []

    async def _rule_based_compliance_check(self, 
                                          text: str,
                                          doc_type: str,
                                          parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform rule-based compliance validation."""
        
        issues = []
        text_lower = text.lower()
        
        if doc_type not in self.compliance_rules:
            return issues
        
        rules = self.compliance_rules[doc_type]
        
        # Check for prohibited terms
        prohibited_terms = rules.get("prohibited_terms", [])
        for term in prohibited_terms:
            if re.search(term, text_lower):
                issues.append({
                    "description": f"Contains prohibited jurisdiction reference: '{term}'",
                    "severity": "High",
                    "section": "Jurisdiction Clause",
                    "adgm_reference": "ADGM Companies Regulations 2020, Art. 6",
                    "suggestion": "Update jurisdiction clause to specify ADGM Courts",
                    "paragraph_index": self._find_paragraph_index(text, term),
                    "highlight_text": term
                })
        
        # Check required patterns
        required_patterns = rules.get("required_patterns", [])
        for pattern in required_patterns:
            if not re.search(pattern, text_lower, re.IGNORECASE):
                pattern_description = self._get_pattern_description(pattern)
                issues.append({
                    "description": f"Missing required clause: {pattern_description}",
                    "severity": "High" if "jurisdiction" in pattern_description.lower() else "Medium",
                    "section": pattern_description,
                    "adgm_reference": self._get_adgm_reference_for_pattern(pattern),
                    "suggestion": f"Add compliant {pattern_description.lower()} clause",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        # Document-specific validations
        if doc_type == "Articles of Association":
            issues.extend(self._validate_articles_of_association(text, parsed_data))
        elif doc_type == "Memorandum of Association":
            issues.extend(self._validate_memorandum_of_association(text, parsed_data))
        elif "Resolution" in doc_type:
            issues.extend(self._validate_resolution_document(text, parsed_data))
        elif doc_type == "UBO Declaration Form":
            issues.extend(self._validate_ubo_declaration(text, parsed_data))
        
        return issues

    def _validate_articles_of_association(self, text: str, parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate Articles of Association specific requirements."""
        
        issues = []
        text_lower = text.lower()
        
        # Check for share capital specification
        if not re.search(r"share\s+capital.*\d+", text_lower):
            issues.append({
                "description": "Share capital amount not clearly specified",
                "severity": "Medium",
                "section": "Share Capital",
                "adgm_reference": "ADGM Companies Regulations 2020, Art. 15",
                "suggestion": "Specify the authorized share capital amount",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        
        # Check for directors' powers
        if not re.search(r"directors?.*powers?", text_lower):
            issues.append({
                "description": "Directors' powers not adequately defined",
                "severity": "Medium", 
                "section": "Directors Powers",
                "adgm_reference": "ADGM Companies Regulations 2020, Art. 45",
                "suggestion": "Include comprehensive directors' powers clause",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        
        # Check structure if available
        if parsed_data and "structure" in parsed_data:
            structure = parsed_data["structure"]
            
            # Check for numbered articles structure
            if len(structure.get("numbered_items", [])) < 5:
                issues.append({
                    "description": "Insufficient article numbering structure",
                    "severity": "Low",
                    "section": "Document Structure", 
                    "adgm_reference": "ADGM Companies Regulations 2020",
                    "suggestion": "Use proper article numbering (1, 2, 3, etc.)",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        return issues

    def _validate_memorandum_of_association(self, text: str, parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate Memorandum of Association specific requirements."""
        
        issues = []
        text_lower = text.lower()
        
        # Check for the five essential clauses
        essential_clauses = {
            "name clause": r"name.*company",
            "objects clause": r"objects?.*company", 
            "liability clause": r"liability.*members?.*limited",
            "capital clause": r"capital.*company.*divided",
            "association clause": r"desire.*formed.*company"
        }
        
        for clause_name, pattern in essential_clauses.items():
            if not re.search(pattern, text_lower):
                issues.append({
                    "description": f"Missing or unclear {clause_name}",
                    "severity": "High",
                    "section": clause_name.title(),
                    "adgm_reference": "ADGM Companies Regulations 2020, Schedule 1",
                    "suggestion": f"Include proper {clause_name} as per ADGM requirements",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        return issues

    def _validate_resolution_document(self, text: str, parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate Resolution document requirements."""
        
        issues = []
        text_lower = text.lower()
        
        # Check for essential resolution elements
        required_elements = {
            "meeting details": r"meeting.*held.*date",
            "quorum": r"quorum.*present",
            "resolution text": r"resolved\s+that",
            "voting": r"unanimously|majority|voted"
        }
        
        for element, pattern in required_elements.items():
            if not re.search(pattern, text_lower):
                issues.append({
                    "description": f"Missing {element} in resolution",
                    "severity": "Medium",
                    "section": element.title(),
                    "adgm_reference": "ADGM Companies Regulations 2020, Art. 32",
                    "suggestion": f"Include proper {element} section",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        # Check for signature section
        if parsed_data and "structure" in parsed_data:
            signatures = parsed_data["structure"].get("signatures", [])
            if len(signatures) == 0:
                issues.append({
                    "description": "No signature section found",
                    "severity": "High",
                    "section": "Signatures",
                    "adgm_reference": "ADGM Companies Regulations 2020",
                    "suggestion": "Add proper signature blocks for directors",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        return issues

    def _validate_ubo_declaration(self, text: str, parsed_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate UBO Declaration specific requirements."""
        
        issues = []
        text_lower = text.lower()
        
        # Check for percentage specifications
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        if not percentages:
            issues.append({
                "description": "No ownership percentages specified",
                "severity": "High",
                "section": "Ownership Details",
                "adgm_reference": "ADGM AML Rules 2020, Rule 3.2",
                "suggestion": "Specify exact ownership percentages for all beneficial owners",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        else:
            # Validate percentage totals
            total_percentage = sum(float(p.replace('%', '')) for p in percentages)
            if total_percentage > 100:
                issues.append({
                    "description": f"Ownership percentages exceed 100% (total: {total_percentage}%)",
                    "severity": "Critical",
                    "section": "Ownership Calculation",
                    "adgm_reference": "ADGM AML Rules 2020",
                    "suggestion": "Ensure ownership percentages total exactly 100%",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        return issues

    async def _rag_powered_compliance_check(self, 
                                           text: str,
                                           doc_type: str) -> List[Dict[str, Any]]:
        """Use RAG to find specific ADGM regulation violations."""
        
        if not self.rag_engine:
            return []
        
        issues = []
        
        try:
            # Query for document-specific regulations
            regulation_query = f"ADGM {doc_type} requirements regulations compliance"
            
            relevant_regulations = await self.rag_engine.hierarchical_retrieve(
                query=regulation_query,
                document_type=doc_type,
                max_results=5
            )
            
            # Analyze each regulation against document content
            for regulation in relevant_regulations:
                reg_content = regulation.get("content", "")
                reg_source = regulation.get("metadata", {}).get("source", "ADGM Regulation")
                
                # Use AI to compare document against specific regulation
                compliance_issue = await self._check_against_specific_regulation(
                    text, reg_content, reg_source, doc_type
                )
                
                if compliance_issue:
                    issues.append(compliance_issue)
            
        except Exception as e:
            logger.error(f"RAG compliance check failed: {str(e)}")
        
        return issues

    async def _check_against_specific_regulation(self, 
                                                document_text: str,
                                                regulation_text: str,
                                                regulation_source: str,
                                                doc_type: str) -> Optional[Dict[str, Any]]:
        """Check document compliance against a specific regulation."""
        
        prompt = f"""
You are an ADGM compliance expert. Compare this document against the specific regulation and identify any violations.

Document Type: {doc_type}
Regulation Source: {regulation_source}

Regulation Text:
{regulation_text[:1000]}

Document Text (excerpt):
{document_text[:1500]}

If you find a compliance violation, respond with JSON:
{{
    "violation_found": true,
    "description": "specific compliance issue",
    "severity": "Critical|High|Medium|Low", 
    "section": "affected document section",
    "suggestion": "how to fix the issue"
}}

If no violation, respond with:
{{"violation_found": false}}

Focus on ADGM-specific requirements only.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ADGM compliance analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            result = json.loads(result_text)
            
            if result.get("violation_found"):
                return {
                    "description": result.get("description", "Compliance violation found"),
                    "severity": result.get("severity", "Medium"),
                    "section": result.get("section", "Unknown"),
                    "adgm_reference": regulation_source,
                    "suggestion": result.get("suggestion", "Review against ADGM requirements"),
                    "paragraph_index": 0,
                    "highlight_text": ""
                }
            
        except Exception as e:
            logger.error(f"Regulation check failed: {str(e)}")
        
        return None

    async def _ai_powered_compliance_analysis(self, 
                                             text: str,
                                             doc_type: str) -> List[Dict[str, Any]]:
        """Use AI for comprehensive compliance analysis."""
        
        prompt = f"""
You are an expert ADGM legal compliance analyst. Analyze this {doc_type} for compliance issues.

Document excerpt:
{text[:2000]}

Check for these ADGM-specific issues:
1. Jurisdiction clauses (must specify ADGM Courts)
2. Registered office requirements (must be in ADGM)
3. Missing mandatory clauses for {doc_type}
4. Template compliance issues
5. Regulatory filing requirements

Return JSON array of issues:
[
  {{
    "description": "specific issue description",
    "severity": "Critical|High|Medium|Low",
    "section": "document section affected", 
    "suggestion": "specific fix recommendation",
    "adgm_reference": "relevant ADGM regulation"
  }}
]

Maximum 5 most critical issues. Empty array if compliant.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ADGM compliance analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            issues = json.loads(result_text)
            
            # Convert to our format
            formatted_issues = []
            for issue in issues:
                formatted_issues.append({
                    "description": issue.get("description", ""),
                    "severity": issue.get("severity", "Medium"),
                    "section": issue.get("section", "General"),
                    "adgm_reference": issue.get("adgm_reference", "ADGM Regulations"),
                    "suggestion": issue.get("suggestion", ""),
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"AI compliance analysis failed: {str(e)}")
            return []

    def _deduplicate_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate issues and prioritize by severity."""
        
        # Group similar issues
        unique_issues = {}
        severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        
        for issue in issues:
            # Create key based on description similarity
            key = issue["description"][:50].lower()
            
            if key not in unique_issues:
                unique_issues[key] = issue
            else:
                # Keep higher severity issue
                existing_severity = severity_order.get(unique_issues[key]["severity"], 0)
                new_severity = severity_order.get(issue["severity"], 0)
                
                if new_severity > existing_severity:
                    unique_issues[key] = issue
        
        # Sort by severity
        final_issues = list(unique_issues.values())
        final_issues.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
        
        return final_issues

    def _find_paragraph_index(self, text: str, search_term: str) -> int:
        """Find paragraph index containing the search term."""
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if search_term.lower() in paragraph.lower():
                return i
        
        return 0

    def _get_pattern_description(self, pattern: str) -> str:
        """Get human-readable description for regex pattern."""
        
        descriptions = {
            r"adgm\s+courts?\s+shall\s+have\s+jurisdiction": "ADGM Jurisdiction Clause",
            r"registered\s+office.*adgm": "ADGM Registered Office",
            r"share\s+capital": "Share Capital Specification",
            r"objects?\s+of\s+the\s+company": "Company Objects Clause",
            r"liability.*limited": "Liability Limitation Clause",
            r"quorum.*present": "Quorum Confirmation",
            r"resolved\s+that": "Resolution Statement"
        }
        
        return descriptions.get(pattern, "Required Clause")

    def _get_adgm_reference_for_pattern(self, pattern: str) -> str:
        """Get ADGM regulation reference for pattern."""
        
        references = {
            r"adgm\s+courts?\s+shall\s+have\s+jurisdiction": "ADGM Companies Regulations 2020, Art. 6",
            r"registered\s+office.*adgm": "ADGM Companies Regulations 2020, Art. 12",
            r"share\s+capital": "ADGM Companies Regulations 2020, Art. 15",
            r"objects?\s+of\s+the\s+company": "ADGM Companies Regulations 2020, Schedule 1",
            r"liability.*limited": "ADGM Companies Regulations 2020, Schedule 1",
            r"quorum.*present": "ADGM Companies Regulations 2020, Art. 32",
            r"resolved\s+that": "ADGM Companies Regulations 2020, Art. 32"
        }
        
        return references.get(pattern, "ADGM Companies Regulations 2020")

    async def validate_against_template(self, 
                                       document_text: str,
                                       document_type: str) -> List[Dict[str, Any]]:
        """Validate document against official ADGM templates."""
        
        if not self.rag_engine:
            return []
        
        # Retrieve official template if available
        template_query = f"ADGM official template {document_type}"
        template_results = await self.rag_engine.hierarchical_retrieve(
            query=template_query,
            document_type="templates",
            max_results=1
        )
        
        if not template_results:
            return []
        
        template_content = template_results[0].get("content", "")
        
        # Compare structure and key elements
        return await self._compare_with_template(document_text, template_content, document_type)

    async def _compare_with_template(self, 
                                    document_text: str,
                                    template_text: str,
                                    doc_type: str) -> List[Dict[str, Any]]:
        """Compare document with official template."""
        
        prompt = f"""
Compare this {doc_type} with the official ADGM template. Identify deviations.

Official Template (excerpt):
{template_text[:1000]}

Submitted Document (excerpt):
{document_text[:1000]}

Return JSON array of template compliance issues:
[
  {{
    "description": "specific deviation from template",
    "severity": "Medium|Low",
    "suggestion": "how to align with template"
  }}
]

Focus on structure, required sections, and formatting requirements.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ADGM template compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            issues = json.loads(result_text)
            
            # Format issues
            formatted_issues = []
            for issue in issues:
                formatted_issues.append({
                    "description": f"Template deviation: {issue.get('description', '')}",
                    "severity": issue.get("severity", "Low"),
                    "section": "Document Structure",
                    "adgm_reference": f"ADGM Official {doc_type} Template",
                    "suggestion": issue.get("suggestion", ""),
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"Template comparison failed: {str(e)}")
            return []

    def get_compliance_stats(self) -> Dict[str, Any]:
        """Get compliance checking statistics."""
        
        return {
            "supported_document_types": len(self.compliance_rules),
            "total_compliance_rules": sum(len(rules) for rules in self.compliance_rules.values()),
            "rag_enabled": self.rag_engine is not None,
            "ai_analysis_enabled": True
        }
 "