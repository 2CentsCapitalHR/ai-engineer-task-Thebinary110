"""
Risk Assessment Agent
Identifies and scores compliance risks and red flags.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI

from adgm_corporate_agent.utils.logger import setup_logger
from adgm_corporate_agent.utils.cache_manager import get_cache_manager

logger = setup_logger(__name__)
cache = get_cache_manager()

@dataclass
class RiskAssessment:
    """Risk assessment result for a document."""
    overall_risk_score: float  # 0-100
    risk_level: str  # Low, Medium, High, Critical
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float

class RiskAssessorAgent:
    """
    Advanced risk assessment agent for ADGM compliance.
    Identifies red flags, calculates risk scores, and provides mitigation recommendations.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the risk assessor."""
        
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.rag_engine = None
        
        # Risk factors with weights and scoring
        self.risk_factors = {
            "jurisdiction_issues": {
                "weight": 25,
                "patterns": [
                    r"uae federal court",
                    r"dubai court", 
                    r"abu dhabi court",
                    r"sharjah court"
                ],
                "description": "Incorrect jurisdiction references"
            },
            
            "missing_mandatory_clauses": {
                "weight": 20,
                "critical_clauses": {
                    "Articles of Association": [
                        "jurisdiction clause",
                        "registered office", 
                        "share capital",
                        "directors powers"
                    ],
                    "Memorandum of Association": [
                        "name clause",
                        "objects clause",
                        "liability clause",
                        "capital clause"
                    ]
                },
                "description": "Absence of mandatory legal clauses"
            },
            
            "template_non_compliance": {
                "weight": 15,
                "indicators": [
                    "non_standard_formatting",
                    "missing_sections",
                    "incorrect_structure"
                ],
                "description": "Deviations from ADGM templates"
            },
            
            "ambiguous_language": {
                "weight": 15,
                "patterns": [
                    r"may\s+be\s+interpreted",
                    r"subject\s+to\s+interpretation",
                    r"at\s+the\s+discretion",
                    r"if\s+deemed\s+appropriate"
                ],
                "description": "Vague or non-binding language"
            },
            
            "signature_deficiencies": {
                "weight": 10,
                "issues": [
                    "missing_signature_blocks",
                    "incomplete_witness_sections",
                    "missing_dates",
                    "unclear_signatory_capacity"
                ],
                "description": "Signature and execution issues"
            },
            
            "inconsistent_information": {
                "weight": 10,
                "checks": [
                    "name_consistency",
                    "address_consistency", 
                    "date_consistency",
                    "reference_consistency"
                ],
                "description": "Conflicting information within document"
            },
            
            "regulatory_gaps": {
                "weight": 5,
                "areas": [
                    "anti_money_laundering",
                    "data_protection",
                    "employment_law",
                    "corporate_governance"
                ],
                "description": "Missing regulatory compliance elements"
            }
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            "Low": (0, 25),
            "Medium": (25, 50),
            "High": (50, 75),
            "Critical": (75, 100)
        }
        
        logger.info("Risk assessor initialized")

    async def initialize(self, rag_engine):
        """Initialize with RAG engine for regulation retrieval."""
        self.rag_engine = rag_engine
        logger.info("Risk assessor connected to RAG engine")

    async def assess_risks(self, 
                          document_text: str,
                          document_type: str,
                          classification_confidence: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform comprehensive risk assessment on document.
        
        Args:
            document_text: Full document text
            document_type: Classified document type
            classification_confidence: Confidence in document classification
            
        Returns:
            List of risk issues found
        """
        
        # Check cache first
        cache_key = f"risk_assessment_{hash(document_text[:1000])}_{document_type}"
        cached_risks = cache.get(cache_key)
        if cached_risks:
            logger.info("Using cached risk assessment")
            return cached_risks
        
        try:
            logger.info(f"Starting risk assessment for {document_type}")
            
            # Step 1: Pattern-based risk detection
            pattern_risks = await self._pattern_based_risk_detection(
                document_text, document_type
            )
            
            # Step 2: AI-powered risk analysis
            ai_risks = await self._ai_powered_risk_analysis(
                document_text, document_type, classification_confidence
            )
            
            # Step 3: Structural risk assessment
            structural_risks = await self._structural_risk_assessment(
                document_text, document_type
            )
            
            # Step 4: Cross-reference risk validation
            if self.rag_engine:
                cross_ref_risks = await self._cross_reference_risk_validation(
                    document_text, document_type
                )
            else:
                cross_ref_risks = []
            
            # Combine all risks
            all_risks = pattern_risks + ai_risks + structural_risks + cross_ref_risks
            
            # Filter and prioritize
            final_risks = self._prioritize_risks(all_risks)
            
            # Cache results
            cache.set(cache_key, final_risks, ttl=1800)  # 30 minutes
            
            logger.info(f"Identified {len(final_risks)} risk factors")
            return final_risks
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return []

    async def _pattern_based_risk_detection(self, 
                                           text: str,
                                           doc_type: str) -> List[Dict[str, Any]]:
        """Detect risks using predefined patterns."""
        
        risks = []
        text_lower = text.lower()
        
        # Check jurisdiction issues
        jurisdiction_patterns = self.risk_factors["jurisdiction_issues"]["patterns"]
        for pattern in jurisdiction_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                risks.append({
                    "description": f"Incorrect jurisdiction reference found: {matches[0]}",
                    "severity": "High",
                    "risk_type": "jurisdiction_issues",
                    "risk_score": 80,
                    "section": "Jurisdiction Clause",
                    "suggestion": "Update to specify ADGM Courts jurisdiction",
                    "paragraph_index": self._find_paragraph_with_text(text, matches[0]),
                    "highlight_text": matches[0]
                })
        
        # Check for ambiguous language
        ambiguous_patterns = self.risk_factors["ambiguous_language"]["patterns"]
        for pattern in ambiguous_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                risks.append({
                    "description": f"Ambiguous language detected: '{matches[0]}'",
                    "severity": "Medium",
                    "risk_type": "ambiguous_language",
                    "risk_score": 45,
                    "section": "Language Clarity",
                    "suggestion": "Replace with definitive, binding language",
                    "paragraph_index": self._find_paragraph_with_text(text, matches[0]),
                    "highlight_text": matches[0]
                })
        
        # Check mandatory clauses for document type
        if doc_type in self.risk_factors["missing_mandatory_clauses"]["critical_clauses"]:
            missing_clauses = self._check_mandatory_clauses(text, doc_type)
            for clause in missing_clauses:
                risks.append({
                    "description": f"Missing mandatory clause: {clause}",
                    "severity": "High",
                    "risk_type": "missing_mandatory_clauses",
                    "risk_score": 70,
                    "section": clause,
                    "suggestion": f"Add compliant {clause.lower()} to document",
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
        
        return risks

    async def _ai_powered_risk_analysis(self, 
                                       text: str,
                                       doc_type: str,
                                       classification_confidence: float) -> List[Dict[str, Any]]:
        """Use AI to identify complex risk patterns."""
        
        prompt = f"""
You are an expert ADGM compliance risk analyst. Analyze this {doc_type} for compliance risks.

Document Classification Confidence: {classification_confidence:.2%}

Document excerpt:
{text[:2000]}

Identify risks in these categories:
1. Legal enforceability issues
2. Regulatory compliance gaps  
3. Corporate governance weaknesses
4. Operational risks
5. Documentation deficiencies

For each risk, assess:
- Severity (Critical/High/Medium/Low)
- Likelihood of regulatory challenge
- Potential consequences

Return JSON array:
[
  {{
    "description": "specific risk description",
    "severity": "Critical|High|Medium|Low",
    "risk_score": 0-100,
    "risk_category": "legal|regulatory|governance|operational|documentation",
    "likelihood": "probability of issue arising",
    "consequences": "potential impact if realized",
    "mitigation": "recommended mitigation strategy"
  }}
]

Maximum 6 most significant risks. Focus on ADGM-specific requirements.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ADGM compliance risk analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            ai_risks = json.loads(result_text)
            
            # Convert to our format
            formatted_risks = []
            for risk in ai_risks:
                formatted_risks.append({
                    "description": risk.get("description", ""),
                    "severity": risk.get("severity", "Medium"),
                    "risk_type": f"ai_detected_{risk.get('risk_category', 'general')}",
                    "risk_score": risk.get("risk_score", 50),
                    "section": risk.get("risk_category", "General").title(),
                    "suggestion": risk.get("mitigation", ""),
                    "likelihood": risk.get("likelihood", ""),
                    "consequences": risk.get("consequences", ""),
                    "paragraph_index": 0,
                    "highlight_text": ""
                })
            
            return formatted_risks
            
        except Exception as e:
            logger.error(f"AI risk analysis failed: {str(e)}")
            return []

    async def _structural_risk_assessment(self, 
                                         text: str,
                                         doc_type: str) -> List[Dict[str, Any]]:
        """Assess structural and formatting risks."""
        
        risks = []
        
        # Document length assessment
        word_count = len(text.split())
        if word_count < 100:
            risks.append({
                "description": f"Document appears too short ({word_count} words) for {doc_type}",
                "severity": "Medium",
                "risk_type": "structural_inadequacy",
                "risk_score": 55,
                "section": "Document Length",
                "suggestion": "Ensure all required sections are included",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        
        # Signature block assessment
        signature_indicators = [
            "signature", "signed by", "director", "secretary", 
            "witness", "date", "__________"
        ]
        
        signature_count = sum(1 for indicator in signature_indicators 
                            if indicator in text.lower())
        
        if signature_count < 2 and "resolution" in doc_type.lower():
            risks.append({
                "description": "Insufficient signature provisions for resolution document",
                "severity": "High", 
                "risk_type": "signature_deficiencies",
                "risk_score": 65,
                "section": "Execution",
                "suggestion": "Add proper signature blocks with witness provisions",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        
        # Date consistency check
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}', text)
        if len(set(dates)) > 3:
            risks.append({
                "description": f"Multiple inconsistent dates found ({len(set(dates))} different dates)",
                "severity": "Medium",
                "risk_type": "inconsistent_information", 
                "risk_score": 40,
                "section": "Date Consistency",
                "suggestion": "Verify and align all document dates",
                "paragraph_index": 0,
                "highlight_text": ""
            })
        
        return risks

    async def _cross_reference_risk_validation(self, 
                                              text: str,
                                              doc_type: str) -> List[Dict[str, Any]]:
        """Validate risks against ADGM regulation database."""
        
        if not self.rag_engine:
            return []
        
        risks = []
        
        try:
            # Query for risk-related regulations
            risk_query = f"ADGM {doc_type} compliance risks violations penalties"
            
            relevant_regulations = await self.rag_engine.hierarchical_retrieve(
                query=risk_query,
                document_type=doc_type,
                max_results=3
            )
            
            for regulation in relevant_regulations:
                reg_content = regulation.get("content", "")
                
                # Check if document violates specific regulation
                violation_risk = await self._assess_regulation_violation_risk(
                    text, reg_content, doc_type
                )
                
                if violation_risk:
                    risks.append(violation_risk)
        
        except Exception as e:
            logger.error(f"Cross-reference risk validation failed: {str(e)}")
        
        return risks

    async def _assess_regulation_violation_risk(self, 
                                               document_text: str,
                                               regulation_text: str,
                                               doc_type: str) -> Optional[Dict[str, Any]]:
        """Assess risk of violating specific regulation."""
        
        prompt = f"""
Assess compliance risk of this {doc_type} against the ADGM regulation.

Regulation:
{regulation_text[:800]}

Document excerpt:
{document_text[:800]}

Evaluate:
1. Risk of regulatory violation (0-100 score)
2. Severity of potential consequences
3. Likelihood of enforcement action

Return JSON if significant risk found:
{{
    "risk_detected": true,
    "risk_score": 0-100,
    "severity": "Critical|High|Medium|Low",
    "description": "specific violation risk",
    "consequences": "potential regulatory consequences",
    "mitigation": "risk mitigation strategy"
}}

Return {{"risk_detected": false}} if no significant risk.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ADGM regulatory risk expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            result = json.loads(result_text)
            
            if result.get("risk_detected"):
                return {
                    "description": f"Regulatory violation risk: {result.get('description', '')}",
                    "severity": result.get("severity", "Medium"),
                    "risk_type": "regulatory_violation",
                    "risk_score": result.get("risk_score", 50),
                    "section": "Regulatory Compliance",
                    "suggestion": result.get("mitigation", ""),
                    "consequences": result.get("consequences", ""),
                    "paragraph_index": 0,
                    "highlight_text": ""
                }
            
        except Exception as e:
            logger.error(f"Regulation violation assessment failed: {str(e)}")
        
        return None

    def _check_mandatory_clauses(self, text: str, doc_type: str) -> List[str]:
        """Check for missing mandatory clauses."""
        
        if doc_type not in self.risk_factors["missing_mandatory_clauses"]["critical_clauses"]:
            return []
        
        required_clauses = self.risk_factors["missing_mandatory_clauses"]["critical_clauses"][doc_type]
        missing_clauses = []
        text_lower = text.lower()
        
        clause_patterns = {
            "jurisdiction clause": r"jurisdiction.*court",
            "registered office": r"registered\s+office",
            "share capital": r"share\s+capital",
            "directors powers": r"directors?.*powers?",
            "name clause": r"name.*company",
            "objects clause": r"objects?.*company",
            "liability clause": r"liability.*limited",
            "capital clause": r"capital.*divided"
        }
        
        for clause in required_clauses:
            pattern = clause_patterns.get(clause.lower(), clause.lower())
            if not re.search(pattern, text_lower):
                missing_clauses.append(clause)
        
        return missing_clauses

    def _find_paragraph_with_text(self, text: str, search_text: str) -> int:
        """Find paragraph index containing specific text."""
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if search_text.lower() in paragraph.lower():
                return i
        
        return 0

    def _prioritize_risks(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and filter risks by severity and score."""
        
        # Remove duplicates based on description similarity
        unique_risks = {}
        for risk in risks:
            key = risk["description"][:30].lower()
            if key not in unique_risks or risk["risk_score"] > unique_risks[key]["risk_score"]:
                unique_risks[key] = risk
        
        # Sort by risk score
        sorted_risks = sorted(unique_risks.values(), 
                            key=lambda x: x["risk_score"], reverse=True)
        
        # Limit to top 10 risks
        return sorted_risks[:10]

    def calculate_overall_risk_score(self, risks: List[Dict[str, Any]]) -> RiskAssessment:
        """Calculate overall risk assessment for document."""
        
        if not risks:
            return RiskAssessment(
                overall_risk_score=10.0,
                risk_level="Low",
                risk_factors=[],
                recommendations=["Document appears compliant with basic requirements"],
                confidence=0.9
            )
        
        # Calculate weighted risk score
        total_score = 0
        total_weight = 0
        risk_categories = {}
        
        for risk in risks:
            risk_type = risk.get("risk_type", "general")
            score = risk.get("risk_score", 50)
            weight = self.risk_factors.get(risk_type, {}).get("weight", 5)
            
            total_score += score * weight
            total_weight += weight
            
            if risk_type not in risk_categories:
                risk_categories[risk_type] = []
            risk_categories[risk_type].append(risk)
        
        # Calculate overall score
        overall_score = (total_score / total_weight) if total_weight > 0 else 0
        
        # Determine risk level
        risk_level = "Low"
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= overall_score < max_score:
                risk_level = level
                break
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(risks, risk_categories)
        
        # Calculate confidence
        confidence = self._calculate_confidence(risks, overall_score)
        
        return RiskAssessment(
            overall_risk_score=overall_score,
            risk_level=risk_level,
            risk_factors=risks,
            recommendations=recommendations,
            confidence=confidence
        )

    def _generate_risk_recommendations(self, 
                                      risks: List[Dict[str, Any]],
                                      risk_categories: Dict[str, List]) -> List[str]:
        """Generate actionable risk mitigation recommendations."""
        
        recommendations = []
        
        # High-priority recommendations
        critical_risks = [r for r in risks if r.get("severity") == "Critical"]
        high_risks = [r for r in risks if r.get("severity") == "High"]
        
        if critical_risks:
            recommendations.append(
                f"URGENT: Address {len(critical_risks)} critical compliance issues before submission"
            )
        
        if high_risks:
            recommendations.append(
                f"HIGH PRIORITY: Resolve {len(high_risks)} high-risk compliance issues"
            )
        
        # Category-specific recommendations
        if "jurisdiction_issues" in risk_categories:
            recommendations.append(
                "Update all jurisdiction clauses to specify ADGM Courts exclusively"
            )
        
        if "missing_mandatory_clauses" in risk_categories:
            recommendations.append(
                "Add all mandatory clauses as required by ADGM regulations"
            )
        
        if "signature_deficiencies" in risk_categories:
            recommendations.append(
                "Complete signature blocks with proper witness and dating provisions"
            )
        
        if "ambiguous_language" in risk_categories:
            recommendations.append(
                "Replace vague language with definitive, legally binding terms"
            )
        
        # General recommendations
        if len(risks) > 5:
            recommendations.append(
                "Consider comprehensive legal review given multiple compliance issues"
            )
        
        if not recommendations:
            recommendations.append("Document shows good compliance - minor improvements recommended")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _calculate_confidence(self, risks: List[Dict[str, Any]], overall_score: float) -> float:
        """Calculate confidence in risk assessment."""
        
        base_confidence = 0.8
        
        # Reduce confidence for edge cases
        if overall_score < 5 or overall_score > 95:
            base_confidence -= 0.1
        
        # Increase confidence with more risk factors identified
        risk_factor_bonus = min(len(risks) * 0.02, 0.15)
        
        # Reduce confidence if many low-severity risks
        low_severity_count = len([r for r in risks if r.get("severity") == "Low"])
        if low_severity_count > 3:
            base_confidence -= 0.1
        
        final_confidence = max(0.5, min(0.95, base_confidence + risk_factor_bonus))
        return round(final_confidence, 2)

    async def generate_risk_report(self, 
                                  document_text: str,
                                  document_type: str,
                                  classification_confidence: float = 1.0) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report."""
        
        # Perform risk assessment
        risks = await self.assess_risks(document_text, document_type, classification_confidence)
        
        # Calculate overall assessment
        overall_assessment = self.calculate_overall_risk_score(risks)
        
        # Build comprehensive report
        report = {
            "overall_risk_score": overall_assessment.overall_risk_score,
            "risk_level": overall_assessment.risk_level,
            "confidence": overall_assessment.confidence,
            "summary": {
                "total_risks": len(risks),
                "critical_risks": len([r for r in risks if r.get("severity") == "Critical"]),
                "high_risks": len([r for r in risks if r.get("severity") == "High"]),
                "medium_risks": len([r for r in risks if r.get("severity") == "Medium"]),
                "low_risks": len([r for r in risks if r.get("severity") == "Low"])
            },
            "risk_breakdown": self._categorize_risks(risks),
            "top_risks": risks[:5],  # Top 5 highest scoring risks
            "recommendations": overall_assessment.recommendations,
            "mitigation_priority": self._prioritize_mitigation(risks)
        }
        
        return report

    def _categorize_risks(self, risks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize risks by type for better organization."""
        
        categories = {
            "Legal & Compliance": [],
            "Documentation": [],
            "Structural": [],
            "Regulatory": [],
            "Operational": []
        }
        
        category_mapping = {
            "jurisdiction_issues": "Legal & Compliance",
            "missing_mandatory_clauses": "Legal & Compliance", 
            "ambiguous_language": "Documentation",
            "signature_deficiencies": "Documentation",
            "template_non_compliance": "Structural",
            "inconsistent_information": "Documentation",
            "regulatory_violation": "Regulatory",
            "ai_detected_legal": "Legal & Compliance",
            "ai_detected_regulatory": "Regulatory",
            "ai_detected_governance": "Operational",
            "ai_detected_operational": "Operational",
            "ai_detected_documentation": "Documentation"
        }
        
        for risk in risks:
            risk_type = risk.get("risk_type", "general")
            category = category_mapping.get(risk_type, "Legal & Compliance")
            categories[category].append(risk)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _prioritize_mitigation(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize mitigation actions by impact and effort."""
        
        mitigation_priorities = []
        
        for risk in risks:
            # Calculate mitigation priority score
            severity_weight = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
            risk_score = risk.get("risk_score", 50)
            severity = risk.get("severity", "Medium")
            
            # Estimate effort (simple heuristic)
            effort_indicators = {
                "jurisdiction_issues": "Low",  # Simple text change
                "missing_mandatory_clauses": "Medium",  # Need to draft clauses
                "signature_deficiencies": "Low",  # Add signature blocks
                "ambiguous_language": "Medium",  # Rewrite sections
                "template_non_compliance": "High",  # Restructure document
                "inconsistent_information": "Low"  # Correct inconsistencies
            }
            
            effort = effort_indicators.get(risk.get("risk_type", ""), "Medium")
            effort_score = {"Low": 1, "Medium": 2, "High": 3}[effort]
            
            # Priority = (Impact Score) / (Effort Score)
            priority_score = (severity_weight[severity] * risk_score / 100) / effort_score
            
            mitigation_priorities.append({
                "risk": risk,
                "priority_score": priority_score,
                "effort_level": effort,
                "suggested_action": self._get_mitigation_action(risk),
                "timeline": self._estimate_timeline(effort, severity)
            })
        
        # Sort by priority score
        mitigation_priorities.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return mitigation_priorities[:8]  # Top 8 priorities

    def _get_mitigation_action(self, risk: Dict[str, Any]) -> str:
        """Get specific mitigation action for risk."""
        
        risk_type = risk.get("risk_type", "")
        
        actions = {
            "jurisdiction_issues": "Replace with 'ADGM Courts shall have exclusive jurisdiction'",
            "missing_mandatory_clauses": f"Add missing clause: {risk.get('section', 'required clause')}",
            "signature_deficiencies": "Add proper signature blocks with witness provisions",
            "ambiguous_language": "Replace with definitive, binding language",
            "template_non_compliance": "Restructure to match ADGM official template",
            "inconsistent_information": "Review and align all references"
        }
        
        return actions.get(risk_type, risk.get("suggestion", "Review and address issue"))

    def _estimate_timeline(self, effort: str, severity: str) -> str:
        """Estimate timeline for mitigation."""
        
        timelines = {
            ("Low", "Critical"): "Immediate (same day)",
            ("Low", "High"): "1-2 days", 
            ("Low", "Medium"): "2-3 days",
            ("Low", "Low"): "1 week",
            ("Medium", "Critical"): "1-2 days",
            ("Medium", "High"): "3-5 days",
            ("Medium", "Medium"): "1 week",
            ("Medium", "Low"): "2 weeks",
            ("High", "Critical"): "3-5 days",
            ("High", "High"): "1-2 weeks",
            ("High", "Medium"): "2-3 weeks",
            ("High", "Low"): "1 month"
        }
        
        return timelines.get((effort, severity), "1-2 weeks")

    def get_risk_assessment_stats(self) -> Dict[str, Any]:
        """Get risk assessment statistics."""
        
        return {
            "risk_factors_monitored": len(self.risk_factors),
            "ai_analysis_enabled": True,
            "rag_validation_enabled": self.rag_engine is not None,
            "risk_categories": list(self.risk_factors.keys()),
            "severity_levels": ["Critical", "High", "Medium", "Low"],
            "risk_thresholds": self.risk_thresholds
        }