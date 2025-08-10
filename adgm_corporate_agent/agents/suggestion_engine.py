"""
Suggestion Generation Agent
Provides compliance suggestions and alternative clauses.
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
class Suggestion:
    """Represents a compliance suggestion."""
    type: str  # clause_replacement, addition, restructuring, general
    description: str
    implementation: str
    priority: str  # Critical, High, Medium, Low
    adgm_reference: str
    estimated_effort: str  # Low, Medium, High
    paragraph_index: int = 0

class SuggestionEngineAgent:
    """
    Advanced suggestion generation agent for ADGM compliance improvements.
    Provides specific, actionable recommendations with implementation guidance.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the suggestion engine."""
        
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.rag_engine = None
        
        # Pre-defined clause templates for common fixes
        self.clause_templates = {
            "jurisdiction_clause": {
                "template": "The Courts of the Abu Dhabi Global Market shall have exclusive jurisdiction to settle any disputes which may arise out of or in connection with this [Document Type].",
                "variations": [
                    "ADGM Courts shall have exclusive jurisdiction over all matters relating to this document.",
                    "Any disputes arising under this agreement shall be subject to the exclusive jurisdiction of the Abu Dhabi Global Market Courts."
                ]
            },
            
            "registered_office_clause": {
                "template": "The registered office of the Company shall be situated in the Abu Dhabi Global Market, United Arab Emirates, at such address as may be determined by the Board of Directors from time to time.",
                "variations": [
                    "The Company's registered office is located within the Abu Dhabi Global Market jurisdiction."
                ]
            },
            
            "share_capital_clause": {
                "template": "The authorized share capital of the Company is [Amount] divided into [Number] shares of [Currency] [Value] each.",
                "variations": [
                    "The Company's share capital shall be [Amount] comprising [Number] ordinary shares of nominal value [Value] each."
                ]
            },
            
            "directors_powers_clause": {
                "template": "Subject to the provisions of the ADGM Companies Regulations and these Articles, the business and affairs of the Company shall be managed by the Board of Directors who may exercise all such powers of the Company.",
                "variations": [
                    "The Directors shall have full power to manage the Company's business in accordance with ADGM regulations and these Articles."
                ]
            },
            
            "liability_limitation_clause": {
                "template": "The liability of the members is limited to the amount, if any, unpaid on the shares respectively held by them.",
                "variations": [
                    "Each member's liability is limited to the unpaid amount on their respective shares."
                ]
            },
            
            "objects_clause": {
                "template": "The objects for which the Company is established are to carry on the business of [Business Description] and all activities ancillary or incidental thereto, subject to the provisions of ADGM regulations.",
                "variations": [
                    "The Company's principal object is to engage in [Business Activity] and related commercial activities within ADGM regulations."
                ]
            }
        }
        
        # Suggestion categories with priorities
        self.suggestion_categories = {
            "critical_compliance": {
                "priority": "Critical",
                "description": "Essential for ADGM compliance",
                "examples": ["jurisdiction fixes", "mandatory clause additions"]
            },
            "regulatory_enhancement": {
                "priority": "High", 
                "description": "Improves regulatory standing",
                "examples": ["template alignment", "best practice adoption"]
            },
            "documentation_improvement": {
                "priority": "Medium",
                "description": "Enhances document quality",
                "examples": ["clarity improvements", "structure optimization"]
            },
            "preventive_measures": {
                "priority": "Low",
                "description": "Reduces future compliance risks",
                "examples": ["additional protections", "optional enhancements"]
            }
        }
        
        logger.info("Suggestion engine initialized")

    async def initialize(self, rag_engine):
        """Initialize with RAG engine for template retrieval."""
        self.rag_engine = rag_engine
        logger.info("Suggestion engine connected to RAG engine")

    async def generate_suggestions(self, 
                                  document_text: str,
                                  document_type: str,
                                  issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive suggestions for document improvement.
        
        Args:
            document_text: Full document text
            document_type: Classified document type
            issues: List of identified compliance/risk issues
            
        Returns:
            List of actionable suggestions
        """
        
        # Check cache first
        cache_key = f"suggestions_{hash(document_text[:1000])}_{len(issues)}"
        cached_suggestions = cache.get(cache_key)
        if cached_suggestions:
            logger.info("Using cached suggestions")
            return cached_suggestions
        
        try:
            logger.info(f"Generating suggestions for {document_type} with {len(issues)} issues")
            
            all_suggestions = []
            
            # Step 1: Issue-specific suggestions
            issue_suggestions = await self._generate_issue_specific_suggestions(
                document_text, document_type, issues
            )
            all_suggestions.extend(issue_suggestions)
            
            # Step 2: Template-based improvements
            template_suggestions = await self._generate_template_improvements(
                document_text, document_type
            )
            all_suggestions.extend(template_suggestions)
            
            # Step 3: Best practice recommendations
            best_practice_suggestions = await self._generate_best_practice_recommendations(
                document_text, document_type
            )
            all_suggestions.extend(best_practice_suggestions)
            
            # Step 4: AI-powered enhancement suggestions
            ai_suggestions = await self._generate_ai_enhancements(
                document_text, document_type, issues
            )
            all_suggestions.extend(ai_suggestions)
            
            # Step 5: Prioritize and filter suggestions
            final_suggestions = self._prioritize_suggestions(all_suggestions)
            
            # Cache results
            cache.set(cache_key, final_suggestions, ttl=1800)  # 30 minutes
            
            logger.info(f"Generated {len(final_suggestions)} suggestions")
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}")
            return []

    async def _generate_issue_specific_suggestions(self, 
                                                  text: str,
                                                  doc_type: str,
                                                  issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate suggestions to fix specific identified issues."""
        
        suggestions = []
        
        for issue in issues:
            issue_type = issue.get("risk_type", issue.get("section", "")).lower()
            severity = issue.get("severity", "Medium")
            description = issue.get("description", "")
            
            # Generate specific suggestion based on issue type
            if "jurisdiction" in issue_type or "jurisdiction" in description.lower():
                suggestion = self._create_jurisdiction_fix_suggestion(issue)
                suggestions.append(suggestion)
                
            elif "mandatory" in description.lower() or "missing" in description.lower():
                suggestion = self._create_missing_clause_suggestion(issue, doc_type)
                suggestions.append(suggestion)
                
            elif "signature" in issue_type or "signature" in description.lower():
                suggestion = self._create_signature_improvement_suggestion(issue)
                suggestions.append(suggestion)
                
            elif "ambiguous" in description.lower() or "vague" in description.lower():
                suggestion = await self._create_language_clarity_suggestion(issue, text)
                suggestions.append(suggestion)
                
            elif "template" in issue_type or "structure" in description.lower():
                suggestion = await self._create_structure_improvement_suggestion(issue, doc_type)
                suggestions.append(suggestion)
        
        return suggestions

    def _create_jurisdiction_fix_suggestion(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Create suggestion to fix jurisdiction issues."""
        
        return {
            "type": "clause_replacement",
            "description": "Replace incorrect jurisdiction clause with ADGM-compliant version",
            "implementation": self.clause_templates["jurisdiction_clause"]["template"],
            "priority": "Critical",
            "adgm_reference": "ADGM Companies Regulations 2020, Article 6",
            "estimated_effort": "Low",
            "paragraph_index": issue.get("paragraph_index", 0),
            "original_issue": issue.get("description", ""),
            "detailed_steps": [
                "1. Locate the current jurisdiction clause",
                "2. Replace with ADGM-compliant language",
                "3. Ensure consistency throughout document"
            ]
        }

    def _create_missing_clause_suggestion(self, issue: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """Create suggestion to add missing mandatory clauses."""
        
        section = issue.get("section", "").lower()
        
        # Determine appropriate template
        template_key = None
        if "share capital" in section:
            template_key = "share_capital_clause"
        elif "directors" in section:
            template_key = "directors_powers_clause"
        elif "registered office" in section:
            template_key = "registered_office_clause"
        elif "objects" in section:
            template_key = "objects_clause"
        elif "liability" in section:
            template_key = "liability_limitation_clause"
        
        implementation = (
            self.clause_templates.get(template_key, {}).get("template", "Add appropriate clause as per ADGM requirements")
            if template_key else "Add required clause according to ADGM regulations"
        )
        
        return {
            "type": "addition",
            "description": f"Add mandatory {section} clause required for {doc_type}",
            "implementation": implementation,
            "priority": "High",
            "adgm_reference": "ADGM Companies Regulations 2020",
            "estimated_effort": "Medium",
            "paragraph_index": 0,
            "original_issue": issue.get("description", ""),
            "detailed_steps": [
                f"1. Identify appropriate location for {section} clause",
                "2. Insert clause using provided template",
                "3. Customize details as applicable",
                "4. Ensure proper numbering and formatting"
            ]
        }

    def _create_signature_improvement_suggestion(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Create suggestion to improve signature provisions."""
        
        return {
            "type": "addition",
            "description": "Add comprehensive signature and execution provisions",
            "implementation": """
EXECUTED as a deed on the date first written above.

SIGNED by [Name], Director                    )
in the presence of:                           )  ________________________
                                             )  [Director Signature]

Witness Signature: ____________________
Witness Name: _________________________
Witness Address: ______________________
Date: _________________________________

SIGNED by [Name], Company Secretary           )
in the presence of:                           )  ________________________
                                             )  [Secretary Signature]

Witness Signature: ____________________
Witness Name: _________________________
""",
            "priority": "High",
            "adgm_reference": "ADGM Companies Regulations 2020",
            "estimated_effort": "Low",
            "paragraph_index": issue.get("paragraph_index", 0),
            "original_issue": issue.get("description", ""),
            "detailed_steps": [
                "1. Add signature blocks at document end",
                "2. Include witness provisions",
                "3. Ensure proper dating",
                "4. Verify signatory authority"
            ]
        }

    async def _create_language_clarity_suggestion(self, 
                                                 issue: Dict[str, Any],
                                                 text: str) -> Dict[str, Any]:
        """Create suggestion to improve language clarity."""
        
        # Extract the problematic text
        highlight_text = issue.get("highlight_text", "")
        
        if highlight_text:
            # Generate specific alternative language
            improvement = await self._generate_language_improvement(highlight_text)
        else:
            improvement = "Replace with definitive, legally binding language"
        
        return {
            "type": "clause_replacement",
            "description": "Replace ambiguous language with clear, binding terms",
            "implementation": improvement,
            "priority": "Medium",
            "adgm_reference": "ADGM Legal Drafting Best Practices",
            "estimated_effort": "Medium",
            "paragraph_index": issue.get("paragraph_index", 0),
            "original_issue": issue.get("description", ""),
            "detailed_steps": [
                "1. Identify ambiguous or vague terms",
                "2. Replace with definitive language",
                "3. Ensure legal enforceability",
                "4. Review for consistency"
            ]
        }

    async def _generate_language_improvement(self, problematic_text: str) -> str:
        """Generate improved language for problematic text."""
        
        prompt = f"""
Improve this legal language to be more definitive and binding for ADGM compliance:

Original text: "{problematic_text}"

Provide a clear, legally binding alternative that:
1. Removes ambiguity
2. Uses definitive language (shall, will, must vs may, might, could)
3. Is enforceable under ADGM law

Return only the improved text.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ADGM legal language expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Language improvement generation failed: {str(e)}")
            return "Replace with definitive, legally binding language"

    async def _create_structure_improvement_suggestion(self, 
                                                      issue: Dict[str, Any],
                                                      doc_type: str) -> Dict[str, Any]:
        """Create suggestion to improve document structure."""
        
        return {
            "type": "restructuring",
            "description": f"Restructure document to align with ADGM {doc_type} template",
            "implementation": await self._get_structure_template(doc_type),
            "priority": "Medium",
            "adgm_reference": f"ADGM Official {doc_type} Template",
            "estimated_effort": "High",
            "paragraph_index": 0,
            "original_issue": issue.get("description", ""),
            "detailed_steps": [
                "1. Review ADGM official template structure",
                "2. Reorganize content to match template",
                "3. Ensure all required sections are present",
                "4. Apply proper formatting and numbering"
            ]
        }

    async def _get_structure_template(self, doc_type: str) -> str:
        """Get structural template for document type."""
        
        templates = {
            "Articles of Association": """
Suggested Structure:
1. Interpretation and Definitions
2. Share Capital and Variation of Rights
3. Issue of Shares
4. Transfer of Shares
5. Directors
6. Powers of Directors
7. Meetings of Directors
8. Shareholders and Meetings
9. Dividends
10. Accounts and Audit
11. Notices
12. Winding Up
13. Jurisdiction
""",
            
            "Memorandum of Association": """
Suggested Structure:
1. Name Clause
2. Registered Office Clause
3. Objects Clause
4. Liability Clause
5. Capital Clause
6. Association Clause
""",
            
            "Board Resolution": """
Suggested Structure:
1. Meeting Details (Date, Time, Location)
2. Directors Present
3. Quorum Confirmation
4. Matters Discussed
5. Resolutions Passed
6. Voting Results
7. Signatures and Date
"""
        }
        
        return templates.get(doc_type, "Follow ADGM template structure for this document type")

    async def _generate_template_improvements(self, 
                                            text: str,
                                            doc_type: str) -> List[Dict[str, Any]]:
        """Generate suggestions based on ADGM template comparison."""
        
        if not self.rag_engine:
            return []
        
        suggestions = []
        
        try:
            # Retrieve official template
            template_query = f"ADGM official template {doc_type} structure format"
            template_results = await self.rag_engine.hierarchical_retrieve(
                query=template_query,
                document_type="templates",
                max_results=1
            )
            
            if template_results:
                template_content = template_results[0].get("content", "")
                
                # Compare with template
                template_suggestions = await self._compare_with_template_suggestions(
                    text, template_content, doc_type
                )
                suggestions.extend(template_suggestions)
        
        except Exception as e:
            logger.error(f"Template improvement generation failed: {str(e)}")
        
        return suggestions

    async def _compare_with_template_suggestions(self, 
                                               document_text: str,
                                               template_text: str,
                                               doc_type: str) -> List[Dict[str, Any]]:
        """Generate suggestions by comparing with official template."""
        
        prompt = f"""
Compare this {doc_type} with the official ADGM template and suggest improvements.

Official Template Structure:
{template_text[:1000]}

Current Document:
{document_text[:1000]}

Provide specific suggestions for:
1. Missing sections
2. Incorrect formatting
3. Non-standard language
4. Structural improvements

Return JSON array of suggestions:
[
  {{
    "description": "specific improvement needed",
    "implementation": "how to implement",
    "priority": "High|Medium|Low",
    "effort": "Low|Medium|High"
  }}
]

Maximum 4 most important improvements.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ADGM template compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            improvements = json.loads(result_text)
            
            # Convert to our format
            suggestions = []
            for improvement in improvements:
                suggestions.append({
                    "type": "template_improvement",
                    "description": improvement.get("description", ""),
                    "implementation": improvement.get("implementation", ""),
                    "priority": improvement.get("priority", "Medium"),
                    "adgm_reference": f"ADGM Official {doc_type} Template",
                    "estimated_effort": improvement.get("effort", "Medium"),
                    "paragraph_index": 0,
                    "detailed_steps": [
                        "1. Review official ADGM template",
                        "2. Implement suggested changes",
                        "3. Verify compliance with template",
                        "4. Test formatting and structure"
                    ]
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Template comparison failed: {str(e)}")
            return []

    async def _generate_best_practice_recommendations(self, 
                                                    text: str,
                                                    doc_type: str) -> List[Dict[str, Any]]:
        """Generate best practice recommendations."""
        
        suggestions = []
        
        # Document-specific best practices
        if doc_type == "Articles of Association":
            suggestions.extend(self._get_aoa_best_practices(text))
        elif doc_type == "Memorandum of Association":
            suggestions.extend(self._get_moa_best_practices(text))
        elif "Resolution" in doc_type:
            suggestions.extend(self._get_resolution_best_practices(text))
        
        # General best practices
        suggestions.extend(self._get_general_best_practices(text, doc_type))
        
        return suggestions

    def _get_aoa_best_practices(self, text: str) -> List[Dict[str, Any]]:
        """Get Articles of Association best practices."""
        
        suggestions = []
        text_lower = text.lower()
        
        # Check for electronic meetings provision
        if "electronic" not in text_lower and "video" not in text_lower:
            suggestions.append({
                "type": "addition",
                "description": "Add provisions for electronic meetings and remote participation",
                "implementation": "The Board may hold meetings by video conference, telephone, or other electronic means that allow all participants to hear each other simultaneously.",
                "priority": "Low",
                "adgm_reference": "ADGM Companies Regulations 2020, Modern Governance",
                "estimated_effort": "Low",
                "paragraph_index": 0
            })
        
        # Check for delegation powers
        if "delegate" not in text_lower and "committee" not in text_lower:
            suggestions.append({
                "type": "addition", 
                "description": "Include director delegation and committee formation powers",
                "implementation": "The Directors may delegate any of their powers to committees consisting of such number of Directors as they think fit.",
                "priority": "Medium",
                "adgm_reference": "ADGM Companies Regulations 2020, Article 45",
                "estimated_effort": "Low",
                "paragraph_index": 0
            })
        
        return suggestions

    def _get_moa_best_practices(self, text: str) -> List[Dict[str, Any]]:
        """Get Memorandum of Association best practices."""
        
        suggestions = []
        text_lower = text.lower()
        
        # Check for broad objects clause
        if len(re.findall(r'objects?.*company', text_lower)) == 1:
            word_count = len(text_lower.split())
            objects_section = re.search(r'objects?.*?(?=\d+\.|$)', text_lower, re.DOTALL)
            objects_words = len(objects_section.group().split()) if objects_section else 0
            
            if objects_words < 50:
                suggestions.append({
                    "type": "enhancement",
                    "description": "Expand objects clause to include ancillary activities",
                    "implementation": "Add: 'and to undertake all activities ancillary, incidental, or conducive to the attainment of the above objects.'",
                    "priority": "Medium",
                    "adgm_reference": "ADGM Companies Regulations 2020",
                    "estimated_effort": "Low",
                    "paragraph_index": 0
                })
        
        return suggestions

    def _get_resolution_best_practices(self, text: str) -> List[Dict[str, Any]]:
        """Get Resolution document best practices."""
        
        suggestions = []
        text_lower = text.lower()
        
        # Check for proper meeting recording
        if "minutes" not in text_lower and "record" not in text_lower:
            suggestions.append({
                "type": "addition",
                "description": "Add reference to minute recording requirements",
                "implementation": "IT WAS RESOLVED that these resolutions be recorded in the minutes of the Company.",
                "priority": "Medium",
                "adgm_reference": "ADGM Companies Regulations 2020, Article 32",
                "estimated_effort": "Low",
                "paragraph_index": 0
            })
        
        # Check for filing requirements mention
        if "filing" not in text_lower and "register" not in text_lower:
            suggestions.append({
                "type": "addition",
                "description": "Include filing and registration requirements",
                "implementation": "The Company Secretary shall ensure all necessary filings are made with ADGM Registration Authority within the prescribed timeframes.",
                "priority": "Low",
                "adgm_reference": "ADGM Companies Regulations 2020",
                "estimated_effort": "Low", 
                "paragraph_index": 0
            })
        
        return suggestions

    def _get_general_best_practices(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Get general best practices applicable to all documents."""
        
        suggestions = []
        text_lower = text.lower()
        
        # Check for defined terms section
        if len(text.split()) > 500 and "definitions" not in text_lower:
            suggestions.append({
                "type": "addition",
                "description": "Consider adding definitions section for key terms",
                "implementation": "Add a definitions section at the beginning defining key terms like 'ADGM', 'Regulations', 'Company', etc.",
                "priority": "Low",
                "adgm_reference": "ADGM Legal Drafting Guidelines",
                "estimated_effort": "Medium",
                "paragraph_index": 0
            })
        
        # Check for update/amendment provisions
        if "amend" not in text_lower and "variation" not in text_lower and doc_type != "Resolution":
            suggestions.append({
                "type": "addition",
                "description": "Add amendment and variation procedures",
                "implementation": f"This {doc_type} may only be amended by special resolution in accordance with ADGM Companies Regulations.",
                "priority": "Low",
                "adgm_reference": "ADGM Companies Regulations 2020",
                "estimated_effort": "Low",
                "paragraph_index": 0
            })
        
        return suggestions

    async def _generate_ai_enhancements(self, 
                                       text: str,
                                       doc_type: str,
                                       issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate AI-powered enhancement suggestions."""
        
        prompt = f"""
As an ADGM legal expert, analyze this {doc_type} and suggest enhancements beyond basic compliance.

Document excerpt:
{text[:1500]}

Current issues identified: {len(issues)}

Suggest improvements for:
1. Legal protection enhancement
2. Operational efficiency 
3. Future-proofing
4. Risk mitigation
5. Best practice adoption

Return JSON array of enhancement suggestions:
[
  {{
    "description": "enhancement description",
    "implementation": "specific implementation guidance",
    "priority": "High|Medium|Low",
    "benefit": "business/legal benefit",
    "effort": "Low|Medium|High"
  }}
]

Maximum 3 most valuable enhancements.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ADGM corporate lawyer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            enhancements = json.loads(result_text)
            
            # Convert to our format
            suggestions = []
            for enhancement in enhancements:
                suggestions.append({
                    "type": "enhancement",
                    "description": enhancement.get("description", ""),
                    "implementation": enhancement.get("implementation", ""),
                    "priority": enhancement.get("priority", "Low"),
                    "adgm_reference": "ADGM Best Practices",
                    "estimated_effort": enhancement.get("effort", "Medium"),
                    "business_benefit": enhancement.get("benefit", ""),
                    "paragraph_index": 0,
                    "detailed_steps": [
                        "1. Review enhancement applicability",
                        "2. Consult with stakeholders",
                        "3. Implement enhancement",
                        "4. Verify compliance maintained"
                    ]
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"AI enhancement generation failed: {str(e)}")
            return []

    def _prioritize_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize suggestions by impact and urgency."""
        
        # Remove duplicates
        unique_suggestions = {}
        for suggestion in suggestions:
            key = suggestion["description"][:50].lower()
            if key not in unique_suggestions:
                unique_suggestions[key] = suggestion
        
        suggestions = list(unique_suggestions.values())
        
        # Priority scoring
        priority_scores = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        effort_scores = {"Low": 1, "Medium": 2, "High": 3}
        
        for suggestion in suggestions:
            priority = suggestion.get("priority", "Medium")
            effort = suggestion.get("estimated_effort", "Medium")
            
            # Impact/Effort ratio for prioritization
            priority_score = priority_scores.get(priority, 2)
            effort_score = effort_scores.get(effort, 2)
            
            suggestion["priority_score"] = priority_score / effort_score
        
        # Sort by priority score
        suggestions.sort(key=lambda x: x.get("priority_score", 1), reverse=True)
        
        # Limit to top 12 suggestions
        return suggestions[:12]

    async def generate_implementation_plan(self, 
                                         suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive implementation plan for suggestions."""
        
        # Group by priority
        priority_groups = {"Critical": [], "High": [], "Medium": [], "Low": []}
        
        for suggestion in suggestions:
            priority = suggestion.get("priority", "Medium")
            priority_groups[priority].append(suggestion)
        
        # Estimate timelines
        timeline_estimates = {
            "Critical": "Immediate (1-2 days)",
            "High": "Short-term (1 week)",
            "Medium": "Medium-term (2-3 weeks)", 
            "Low": "Long-term (1-2 months)"
        }
        
        # Calculate effort distribution
        effort_distribution = {"Low": 0, "Medium": 0, "High": 0}
        for suggestion in suggestions:
            effort = suggestion.get("estimated_effort", "Medium")
            effort_distribution[effort] += 1
        
        implementation_plan = {
            "total_suggestions": len(suggestions),
            "priority_breakdown": {
                priority: len(items) for priority, items in priority_groups.items() if items
            },
            "effort_distribution": effort_distribution,
            "recommended_sequence": self._create_implementation_sequence(suggestions),
            "timeline_estimates": timeline_estimates,
            "resource_requirements": self._estimate_resource_requirements(suggestions),
            "success_metrics": self._define_success_metrics(suggestions)
        }
        
        return implementation_plan

    def _create_implementation_sequence(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create recommended implementation sequence."""
        
        # Sort by priority and effort
        sorted_suggestions = sorted(suggestions, 
                                  key=lambda x: (
                                      {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}.get(x.get("priority", "Medium"), 2),
                                      -{"Low": 1, "Medium": 2, "High": 3}.get(x.get("estimated_effort", "Medium"), 2)
                                  ), reverse=True)
        
        sequence = []
        for i, suggestion in enumerate(sorted_suggestions[:8], 1):
            sequence.append({
                "step": i,
                "description": suggestion["description"],
                "type": suggestion["type"],
                "priority": suggestion.get("priority", "Medium"),
                "estimated_effort": suggestion.get("estimated_effort", "Medium"),
                "dependencies": self._identify_dependencies(suggestion, sorted_suggestions[:i-1])
            })
        
        return sequence

    def _identify_dependencies(self, 
                              suggestion: Dict[str, Any], 
                              previous_suggestions: List[Dict[str, Any]]) -> List[str]:
        """Identify dependencies between suggestions."""
        
        dependencies = []
        current_type = suggestion.get("type", "")
        current_desc = suggestion.get("description", "").lower()
        
        for prev_suggestion in previous_suggestions:
            prev_type = prev_suggestion.get("type", "")
            prev_desc = prev_suggestion.get("description", "").lower()
            
            # Structural dependencies
            if current_type == "addition" and prev_type == "restructuring":
                dependencies.append(f"Complete restructuring before adding new content")
            
            # Content dependencies  
            if "signature" in current_desc and "clause" in prev_desc:
                dependencies.append("Add clauses before finalizing signatures")
        
        return dependencies

    def _estimate_resource_requirements(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate resource requirements for implementation."""
        
        effort_hours = {"Low": 2, "Medium": 8, "High": 24}
        total_hours = sum(effort_hours.get(s.get("estimated_effort", "Medium"), 8) 
                         for s in suggestions)
        
        return {
            "estimated_total_hours": total_hours,
            "legal_review_required": len([s for s in suggestions if s.get("priority") in ["Critical", "High"]]) > 0,
            "external_consultation_needed": any("template" in s.get("type", "") for s in suggestions),
            "document_restructuring_required": any(s.get("type") == "restructuring" for s in suggestions)
        }

    def _define_success_metrics(self, suggestions: List[Dict[str, Any]]) -> List[str]:
        """Define success metrics for implementation."""
        
        metrics = []
        
        # Compliance metrics
        critical_count = len([s for s in suggestions if s.get("priority") == "Critical"])
        if critical_count > 0:
            metrics.append(f"100% of {critical_count} critical compliance issues resolved")
        
        # Quality metrics
        total_suggestions = len(suggestions)
        metrics.append(f"At least 80% of {total_suggestions} suggestions implemented")
        
        # Risk metrics
        metrics.append("Document risk score reduced below Medium threshold")
        
        # Process metrics
        metrics.append("All mandatory ADGM clauses present and compliant")
        metrics.append("Document structure aligned with ADGM templates")
        
        return metrics

    def get_suggestion_templates(self, document_type: str) -> Dict[str, Any]:
        """Get available suggestion templates for document type."""
        
        relevant_templates = {}
        
        if document_type in ["Articles of Association", "Memorandum of Association"]:
            relevant_templates.update(self.clause_templates)
        
        return {
            "document_type": document_type,
            "available_templates": relevant_templates,
            "suggestion_categories": self.suggestion_categories
        }

    def get_suggestion_stats(self) -> Dict[str, Any]:
        """Get suggestion engine statistics."""
        
        return {
            "available_templates": len(self.clause_templates),
            "suggestion_categories": len(self.suggestion_categories),
            "ai_enhancement_enabled": True,
            "template_comparison_enabled": self.rag_engine is not None,
            "best_practices_included": True
        }