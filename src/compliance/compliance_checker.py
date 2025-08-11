"""
ADGM Compliance Checker - Advanced Legal Compliance Analysis Engine
Performs comprehensive compliance checking against ADGM regulations using RAG
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from ..rag.adgm_knowledge_base import ADGMKnowledgeBase
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ComplianceChecker:
    """
    Advanced compliance checker for ADGM legal documents.
    
    Uses RAG to provide context-aware compliance analysis by:
    1. Retrieving relevant ADGM regulations and templates
    2. Analyzing document content against requirements
    3. Identifying specific compliance gaps and issues
    4. Providing actionable recommendations with legal citations
    """
    
    def __init__(
        self, 
        llm: ChatGoogleGenerativeAI, 
        knowledge_base: ADGMKnowledgeBase
    ):
        self.llm = llm
        self.knowledge_base = knowledge_base
        
        # Compliance rule patterns
        self.compliance_patterns = self._load_compliance_patterns()
        
        # Critical compliance areas
        self.critical_areas = [
            "jurisdiction",
            "governing_law", 
            "mandatory_clauses",
            "document_structure",
            "signature_requirements"
        ]
    
    def _load_compliance_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance patterns for different document types"""
        
        return {
            "jurisdiction_compliance": {
                "required_terms": ["adgm", "abu dhabi global market"],
                "prohibited_terms": ["uae federal courts", "dubai courts", "abu dhabi courts"],
                "severity": "High",
                "regulation": "ADGM Companies Regulations 2020, Article 6"
            },
            "language_compliance": {
                "binding_terms": ["shall", "must", "will", "agrees"],
                "avoid_terms": ["may", "might", "possibly", "perhaps", "could"],
                "severity": "Medium",
                "regulation": "Legal drafting best practices"
            },
            "structure_compliance": {
                "required_sections": ["parties", "terms", "signatures", "date"],
                "signature_keywords": ["signature", "signed", "executed", "witness"],
                "severity": "High", 
                "regulation": "ADGM legal document requirements"
            }
        }
    
    async def check_compliance(
        self, 
        document_content: Dict[str, Any],
        document_type: str,
        process_type: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive compliance checking on a document.
        
        Args:
            document_content: Parsed document content
            document_type: Type of document (e.g., "Articles of Association")
            process_type: Legal process (e.g., "company_incorporation")
            
        Returns:
            Detailed compliance analysis results
        """
        logger.info(f"🔍 Starting compliance check for {document_type}")
        
        try:
            # Step 1: Retrieve relevant legal context using RAG
            legal_context = await self._retrieve_legal_context(
                document_content["text"], 
                document_type, 
                process_type
            )
            
            # Step 2: Perform rule-based compliance checks
            rule_based_results = await self._perform_rule_based_checks(
                document_content["text"], 
                document_type
            )
            
            # Step 3: AI-powered contextual analysis
            ai_analysis = await self._perform_ai_compliance_analysis(
                document_content, 
                document_type, 
                legal_context
            )
            
            # Step 4: Template matching (if applicable)
            template_analysis = await self._check_template_compliance(
                document_content, 
                document_type
            )
            
            # Step 5: Combine and prioritize results
            combined_results = self._combine_compliance_results(
                rule_based_results, 
                ai_analysis, 
                template_analysis, 
                legal_context
            )
            
            logger.info(f"✅ Compliance check completed for {document_type}")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Error during compliance check: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _retrieve_legal_context(
        self, 
        document_text: str, 
        document_type: str, 
        process_type: str
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant legal context using RAG"""
        
        if not self.knowledge_base:
            logger.warning("⚠️ Knowledge base not available, using static context")
            return self._get_static_legal_context(document_type, process_type)
        
        try:
            # Create context-specific query
            query = f"ADGM {document_type} {process_type} legal requirements compliance mandatory clauses"
            
            # Get relevant documents from knowledge base
            context_docs = await self.knowledge_base.retrieve_relevant_context(
                query, 
                document_type, 
                top_k=5
            )
            
            return context_docs
            
        except Exception as e:
            logger.warning(f"⚠️ Error retrieving legal context: {str(e)}")
            return self._get_static_legal_context(document_type, process_type)
    
    def _get_static_legal_context(self, document_type: str, process_type: str) -> List[Dict[str, Any]]:
        """Get static legal context when RAG is not available"""
        
        static_context = [
            {
                "content": "ADGM jurisdiction clause must reference Abu Dhabi Global Market Courts exclusively. Documents must NOT reference UAE Federal Courts or Dubai Courts.",
                "metadata": {"source": "ADGM_Static", "category": "jurisdiction"},
                "relevance_score": 0.9,
                "source": "ADGM Legal Framework",
                "category": "jurisdiction"
            },
            {
                "content": "All ADGM legal documents must use definitive language such as 'shall', 'must', and 'will' instead of ambiguous terms like 'may', 'possibly', or 'perhaps'.",
                "metadata": {"source": "ADGM_Static", "category": "language"},
                "relevance_score": 0.8,
                "source": "ADGM Legal Standards", 
                "category": "language"
            },
            {
                "content": "Required documents for company incorporation include Articles of Association, Memorandum of Association, Board Resolution, and Register of Members and Directors.",
                "metadata": {"source": "ADGM_Static", "category": "company_formation"},
                "relevance_score": 0.8,
                "source": "ADGM Company Formation Guide",
                "category": "company_formation"
            }
        ]
        
        return static_context
    
    async def _perform_rule_based_checks(
        self, 
        document_text: str, 
        document_type: str
    ) -> List[Dict[str, Any]]:
        """Perform rule-based compliance checks"""
        
        issues = []
        text_lower = document_text.lower()
        
        # Check 1: Jurisdiction Compliance
        jurisdiction_issues = self._check_jurisdiction_compliance(text_lower)
        issues.extend(jurisdiction_issues)
        
        # Check 2: Language Compliance
        language_issues = self._check_language_compliance(text_lower)
        issues.extend(language_issues)
        
        # Check 3: Structural Compliance
        structure_issues = self._check_structural_compliance(text_lower, document_type)
        issues.extend(structure_issues)
        
        # Check 4: Document-specific checks
        specific_issues = await self._check_document_specific_rules(text_lower, document_type)
        issues.extend(specific_issues)
        
        return issues
    
    def _check_jurisdiction_compliance(self, text_lower: str) -> List[Dict[str, Any]]:
        """Check jurisdiction and governing law compliance"""
        
        issues = []
        patterns = self.compliance_patterns["jurisdiction_compliance"]
        
        # Check for prohibited jurisdiction references
        for prohibited_term in patterns["prohibited_terms"]:
            if prohibited_term in text_lower:
                issues.append({
                    "type": "jurisdiction_error",
                    "severity": patterns["severity"],
                    "issue": f"References {prohibited_term} instead of ADGM Courts",
                    "suggestion": "Update jurisdiction clause to specify ADGM Courts exclusively",
                    "regulation": patterns["regulation"],
                    "section": "Governing Law",
                    "rule_based": True
                })
        
        # Check for missing ADGM references
        has_adgm_reference = any(term in text_lower for term in patterns["required_terms"])
        if not has_adgm_reference:
            issues.append({
                "type": "missing_adgm_reference",
                "severity": "Medium",
                "issue": "No clear reference to ADGM jurisdiction found",
                "suggestion": "Add explicit ADGM jurisdiction and governing law clause",
                "regulation": patterns["regulation"],
                "section": "Governing Law",
                "rule_based": True
            })
        
        return issues
    
    def _check_language_compliance(self, text_lower: str) -> List[Dict[str, Any]]:
        """Check for proper legal language and binding terms"""
        
        issues = []
        patterns = self.compliance_patterns["language_compliance"]
        
        # Check for weak/ambiguous language
        weak_terms_found = []
        for weak_term in patterns["avoid_terms"]:
            if f" {weak_term} " in text_lower:
                weak_terms_found.append(weak_term)
        
        if weak_terms_found:
            issues.append({
                "type": "ambiguous_language",
                "severity": patterns["severity"],
                "issue": f"Use of ambiguous terms: {', '.join(weak_terms_found)}",
                "suggestion": "Replace ambiguous language with definitive terms (shall, must, will)",
                "regulation": patterns["regulation"],
                "section": "Language Usage",
                "rule_based": True,
                "details": {"weak_terms": weak_terms_found}
            })
        
        # Check for presence of binding language
        has_binding_language = any(term in text_lower for term in patterns["binding_terms"])
        if not has_binding_language:
            issues.append({
                "type": "weak_binding_language",
                "severity": "Medium",
                "issue": "Document lacks strong binding language",
                "suggestion": "Include definitive terms like 'shall', 'must', or 'agrees'",
                "regulation": patterns["regulation"],
                "section": "Legal Language",
                "rule_based": True
            })
        
        return issues
    
    def _check_structural_compliance(self, text_lower: str, document_type: str) -> List[Dict[str, Any]]:
        """Check document structure and required sections"""
        
        issues = []
        patterns = self.compliance_patterns["structure_compliance"]
        
        # Check for signature sections
        has_signatures = any(keyword in text_lower for keyword in patterns["signature_keywords"])
        if not has_signatures:
            issues.append({
                "type": "missing_signatures",
                "severity": patterns["severity"],
                "issue": "No signature section detected",
                "suggestion": "Add proper signature block with date and witness provisions",
                "regulation": patterns["regulation"],
                "section": "Document Execution",
                "rule_based": True
            })
        
        # Check for date provisions
        date_patterns = [r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", r"date:", r"dated"]
        has_date = any(re.search(pattern, text_lower) for pattern in date_patterns)
        if not has_date:
            issues.append({
                "type": "missing_date",
                "severity": "Medium", 
                "issue": "No date or dating provision found",
                "suggestion": "Include execution date and dating provisions",
                "regulation": patterns["regulation"],
                "section": "Document Dating",
                "rule_based": True
            })
        
        return issues
    
    async def _check_document_specific_rules(
        self, 
        text_lower: str, 
        document_type: str
    ) -> List[Dict[str, Any]]:
        """Check document-type specific compliance rules"""
        
        issues = []
        doc_type_lower = document_type.lower()
        
        # Articles of Association specific checks
        if "articles" in doc_type_lower and "association" in doc_type_lower:
            issues.extend(self._check_aoa_specific_rules(text_lower))
        
        # Employment Contract specific checks
        elif "employment" in doc_type_lower or "contract" in doc_type_lower:
            issues.extend(self._check_employment_specific_rules(text_lower))
        
        # Board Resolution specific checks
        elif "resolution" in doc_type_lower or "board" in doc_type_lower:
            issues.extend(self._check_resolution_specific_rules(text_lower))
        
        return issues
    
    def _check_aoa_specific_rules(self, text_lower: str) -> List[Dict[str, Any]]:
        """Articles of Association specific compliance checks"""
        
        issues = []
        
        # Check for share capital provisions
        share_keywords = ["share capital", "authorized capital", "shares"]
        has_share_provisions = any(keyword in text_lower for keyword in share_keywords)
        if not has_share_provisions:
            issues.append({
                "type": "missing_share_capital",
                "severity": "High",
                "issue": "No share capital provisions found",
                "suggestion": "Include authorized share capital and share structure details",
                "regulation": "ADGM Companies Regulations 2020, Article 8",
                "section": "Share Capital",
                "rule_based": True
            })
        
        # Check for director provisions
        director_keywords = ["director", "board", "management"]
        has_director_provisions = any(keyword in text_lower for keyword in director_keywords)
        if not has_director_provisions:
            issues.append({
                "type": "missing_director_provisions",
                "severity": "High",
                "issue": "No director/board provisions found",
                "suggestion": "Include director appointment, powers, and responsibilities",
                "regulation": "ADGM Companies Regulations 2020, Article 12",
                "section": "Directors",
                "rule_based": True
            })
        
        return issues
    
    def _check_employment_specific_rules(self, text_lower: str) -> List[Dict[str, Any]]:
        """Employment Contract specific compliance checks"""
        
        issues = []
        
        # Check for salary provisions
        salary_keywords = ["salary", "compensation", "wages", "remuneration"]
        has_salary = any(keyword in text_lower for keyword in salary_keywords)
        if not has_salary:
            issues.append({
                "type": "missing_salary_provisions",
                "severity": "High",
                "issue": "No salary/compensation provisions found",
                "suggestion": "Include clear salary and compensation details",
                "regulation": "ADGM Employment Regulations 2019, Section 5",
                "section": "Compensation",
                "rule_based": True
            })
        
        # Check for termination provisions
        termination_keywords = ["termination", "notice", "end of service"]
        has_termination = any(keyword in text_lower for keyword in termination_keywords)
        if not has_termination:
            issues.append({
                "type": "missing_termination_provisions",
                "severity": "High",
                "issue": "No termination provisions found",
                "suggestion": "Include termination procedures and notice periods",
                "regulation": "ADGM Employment Regulations 2019, Section 12",
                "section": "Termination",
                "rule_based": True
            })
        
        return issues
    
    def _check_resolution_specific_rules(self, text_lower: str) -> List[Dict[str, Any]]:
        """Board Resolution specific compliance checks"""
        
        issues = []
        
        # Check for quorum provisions
        quorum_keywords = ["quorum", "present", "attendance"]
        has_quorum = any(keyword in text_lower for keyword in quorum_keywords)
        if not has_quorum:
            issues.append({
                "type": "missing_quorum",
                "severity": "High",
                "issue": "No quorum provisions found",
                "suggestion": "Include quorum requirements and confirmation",
                "regulation": "ADGM Companies Regulations 2020, Article 15",
                "section": "Meeting Procedures",
                "rule_based": True
            })
        
        # Check for voting provisions
        voting_keywords = ["resolved", "voted", "unanimous", "majority"]
        has_voting = any(keyword in text_lower for keyword in voting_keywords)
        if not has_voting:
            issues.append({
                "type": "missing_voting_details",
                "severity": "Medium",
                "issue": "No voting details found",
                "suggestion": "Include voting results and resolution details",
                "regulation": "ADGM Companies Regulations 2020, Article 16",
                "section": "Voting",
                "rule_based": True
            })
        
        return issues
    
    async def _perform_ai_compliance_analysis(
        self, 
        document_content: Dict[str, Any], 
        document_type: str, 
        legal_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform AI-powered compliance analysis using retrieved legal context"""
        
        # Prepare context from RAG retrieval
        context_text = "\n\n".join([
            f"Legal Reference: {doc['content'][:500]}..." 
            for doc in legal_context[:3]
        ])
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
        As an ADGM legal compliance expert, analyze this {document_type} for compliance issues.
        
        DOCUMENT CONTENT:
        {document_content['text'][:3000]}...
        
        RELEVANT ADGM LEGAL CONTEXT:
        {context_text}
        
        ANALYSIS REQUIRED:
        1. Identify specific ADGM compliance gaps
        2. Check against relevant regulations and requirements
        3. Highlight critical legal issues
        4. Assess template compliance (if applicable)
        5. Provide specific improvement recommendations
        
        Focus on:
        - ADGM jurisdiction and governing law compliance
        - Mandatory clause requirements
        - Legal language precision
        - Document structure completeness
        - Regulatory compliance gaps
        
        Return analysis in JSON format:
        {{
            "overall_compliance_score": 0-100,
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
        """
        
        try:
            response = await self.llm.ainvoke(analysis_prompt)
            ai_analysis = json.loads(response.content)
            
            logger.info(f"🤖 AI compliance analysis completed with score: {ai_analysis.get('overall_compliance_score', 'N/A')}")
            return ai_analysis
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Could not parse AI analysis response")
            return {
                "overall_compliance_score": 50,
                "critical_issues": [],
                "strengths": [],
                "recommendations": ["Manual legal review recommended"],
                "regulatory_references": [],
                "ai_analysis_error": "Failed to parse LLM response"
            }
        except Exception as e:
            logger.error(f"❌ Error in AI compliance analysis: {str(e)}")
            return {
                "overall_compliance_score": 0,
                "critical_issues": [],
                "strengths": [],
                "recommendations": [],
                "regulatory_references": [],
                "ai_analysis_error": str(e)
            }
    
    async def _check_template_compliance(
        self, 
        document_content: Dict[str, Any], 
        document_type: str
    ) -> Dict[str, Any]:
        """Check compliance against official ADGM templates"""
        
        # Get template requirements from knowledge base
        template_type = self._map_document_to_template_type(document_type)
        
        if self.knowledge_base:
            try:
                template_requirements = await self.knowledge_base.get_template_requirements(template_type)
            except Exception as e:
                logger.warning(f"⚠️ Error getting template requirements: {str(e)}")
                template_requirements = self._get_static_template_requirements(template_type)
        else:
            template_requirements = self._get_static_template_requirements(template_type)
        
        if not template_requirements:
            return {"template_compliance": "No template available for comparison"}
        
        text_lower = document_content['text'].lower()
        
        # Check required sections
        missing_sections = []
        for section in template_requirements.get("required_sections", []):
            section_keywords = section.lower().split()
            if not any(keyword in text_lower for keyword in section_keywords):
                missing_sections.append(section)
        
        # Check mandatory clauses
        missing_clauses = []
        for clause in template_requirements.get("mandatory_clauses", []):
            clause_keywords = clause.lower().split()
            if not any(keyword in text_lower for keyword in clause_keywords):
                missing_clauses.append(clause)
        
        # Calculate template compliance score
        total_requirements = (
            len(template_requirements.get("required_sections", [])) +
            len(template_requirements.get("mandatory_clauses", []))
        )
        
        missing_count = len(missing_sections) + len(missing_clauses)
        compliance_score = max(0, (total_requirements - missing_count) / total_requirements * 100) if total_requirements > 0 else 100
        
        return {
            "template_type": template_type,
            "compliance_score": compliance_score,
            "missing_sections": missing_sections,
            "missing_clauses": missing_clauses,
            "template_requirements": template_requirements
        }
    
    def _get_static_template_requirements(self, template_type: str) -> Dict[str, Any]:
        """Get static template requirements when knowledge base is not available"""
        
        static_templates = {
            "articles_of_association": {
                "required_sections": [
                    "Company Name and Registration",
                    "Share Capital Structure", 
                    "Directors and Management",
                    "Shareholder Rights",
                    "Governing Law and Jurisdiction",
                    "Signature and Execution"
                ],
                "mandatory_clauses": [
                    "ADGM jurisdiction clause",
                    "Company objects and purposes",
                    "Share transfer provisions",
                    "Director appointment procedures"
                ],
                "red_flags": [
                    "Non-ADGM jurisdiction references",
                    "Missing share capital details",
                    "Ambiguous director powers",
                    "Missing signature sections"
                ]
            },
            "employment_contract": {
                "required_sections": [
                    "Employee Details",
                    "Job Description and Duties",
                    "Salary and Benefits",
                    "Working Hours and Leave",
                    "Termination Procedures",
                    "Governing Law"
                ],
                "mandatory_clauses": [
                    "ADGM Employment Regulations compliance",
                    "Notice period specifications",
                    "End of service benefits",
                    "ADGM jurisdiction clause"
                ],
                "red_flags": [
                    "Below minimum wage provisions",
                    "Excessive working hours",
                    "Missing termination procedures",
                    "Non-ADGM jurisdiction"
                ]
            },
            "board_resolution": {
                "required_sections": [
                    "Meeting Details",
                    "Attendees and Quorum", 
                    "Resolutions Passed",
                    "Voting Results",
                    "Signature and Authentication"
                ],
                "mandatory_clauses": [
                    "Meeting date and time",
                    "Quorum confirmation",
                    "Resolution text",
                    "Director signatures"
                ],
                "red_flags": [
                    "Missing quorum details",
                    "Ambiguous resolution language",
                    "Missing director signatures",
                    "Incorrect voting procedures"
                ]
            }
        }
        
        return static_templates.get(template_type, {
            "required_sections": [],
            "mandatory_clauses": [],
            "red_flags": []
        })
    
    def _map_document_to_template_type(self, document_type: str) -> str:
        """Map document type to template type for requirements lookup"""
        
        mapping = {
            "articles of association": "articles_of_association",
            "employment contract": "employment_contract",
            "board resolution": "board_resolution",
            "memorandum of association": "memorandum_of_association",
            "shareholder resolution": "shareholder_resolution"
        }
        
        doc_type_lower = document_type.lower()
        for key, value in mapping.items():
            if key in doc_type_lower:
                return value
        
        return "general_legal_document"
    
    def _combine_compliance_results(
        self, 
        rule_based_results: List[Dict[str, Any]], 
        ai_analysis: Dict[str, Any], 
        template_analysis: Dict[str, Any], 
        legal_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine all compliance analysis results into comprehensive report"""
        
        # Combine all issues
        all_issues = []
        
        # Add rule-based issues
        all_issues.extend(rule_based_results)
        
        # Add AI-identified issues
        ai_issues = ai_analysis.get("critical_issues", [])
        for issue in ai_issues:
            issue["ai_generated"] = True
        all_issues.extend(ai_issues)
        
        # Add template compliance issues
        if template_analysis.get("missing_sections"):
            for section in template_analysis["missing_sections"]:
                all_issues.append({
                    "type": "missing_template_section",
                    "severity": "High",
                    "issue": f"Missing required section: {section}",
                    "suggestion": f"Add {section} section as per ADGM template",
                    "section": section,
                    "template_based": True
                })
        
        if template_analysis.get("missing_clauses"):
            for clause in template_analysis["missing_clauses"]:
                all_issues.append({
                    "type": "missing_mandatory_clause",
                    "severity": "High", 
                    "issue": f"Missing mandatory clause: {clause}",
                    "suggestion": f"Include {clause} as required by ADGM",
                    "section": "Mandatory Clauses",
                    "template_based": True
                })
        
        # Calculate overall scores
        severity_weights = {"High": 30, "Medium": 15, "Low": 5}
        total_penalty = sum(severity_weights.get(issue.get("severity", "Medium"), 15) for issue in all_issues)
        
        overall_score = max(0, 100 - total_penalty)
        
        # Determine compliance status
        if overall_score >= 90:
            status = "COMPLIANT"
        elif overall_score >= 70:
            status = "MINOR_ISSUES"
        elif overall_score >= 50:
            status = "NEEDS_REVIEW"
        else:
            status = "MAJOR_ISSUES"
        
        # Generate prioritized recommendations
        recommendations = self._generate_prioritized_recommendations(all_issues, ai_analysis)
        
        return {
            "success": True,
            "overall_compliance_score": overall_score,
            "compliance_status": status,
            "total_issues": len(all_issues),
            "issue_breakdown": {
                "high_priority": sum(1 for issue in all_issues if issue.get("severity") == "High"),
                "medium_priority": sum(1 for issue in all_issues if issue.get("severity") == "Medium"),
                "low_priority": sum(1 for issue in all_issues if issue.get("severity") == "Low")
            },
            "detailed_issues": all_issues,
            "ai_analysis_summary": {
                "score": ai_analysis.get("overall_compliance_score", 0),
                "strengths": ai_analysis.get("strengths", []),
                "regulatory_references": ai_analysis.get("regulatory_references", [])
            },
            "template_compliance": template_analysis,
            "prioritized_recommendations": recommendations,
            "legal_context_used": len(legal_context),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_prioritized_recommendations(
        self, 
        all_issues: List[Dict[str, Any]], 
        ai_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate prioritized recommendations based on all findings"""
        
        recommendations = []
        
        # High priority recommendations from critical issues
        high_priority_issues = [issue for issue in all_issues if issue.get("severity") == "High"]
        for issue in high_priority_issues[:3]:  # Top 3 high priority
            recommendations.append({
                "priority": "High",
                "recommendation": issue.get("suggestion", "Address critical compliance issue"),
                "rationale": f"Critical issue: {issue.get('issue', 'Unknown issue')}",
                "regulation": issue.get("regulation", "ADGM regulations")
            })
        
        # Medium priority recommendations
        medium_priority_issues = [issue for issue in all_issues if issue.get("severity") == "Medium"]
        for issue in medium_priority_issues[:2]:  # Top 2 medium priority
            recommendations.append({
                "priority": "Medium",
                "recommendation": issue.get("suggestion", "Address compliance issue"),
                "rationale": f"Important issue: {issue.get('issue', 'Unknown issue')}",
                "regulation": issue.get("regulation", "ADGM regulations")
            })
        
        # Add AI-generated recommendations
        ai_recommendations = ai_analysis.get("recommendations", [])
        for rec in ai_recommendations[:2]:  # Top 2 AI recommendations
            recommendations.append({
                "priority": "Medium",
                "recommendation": rec,
                "rationale": "AI-identified improvement opportunity",
                "regulation": "Various ADGM regulations"
            })
        
        # General recommendations if no specific issues
        if not recommendations:
            recommendations.extend([
                {
                    "priority": "Low",
                    "recommendation": "Review document for ADGM compliance best practices",
                    "rationale": "Ensure full regulatory compliance",
                    "regulation": "ADGM legal framework"
                },
                {
                    "priority": "Low", 
                    "recommendation": "Consider legal review before finalization",
                    "rationale": "Professional validation recommended",
                    "regulation": "Professional legal standards"
                }
            ])
        
        return recommendations