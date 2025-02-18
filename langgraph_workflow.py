# langgraph_workflow.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Dict, Any, Tuple, Optional
from config import Config, load_config
from llm_interface import LLMInterface
from arxiv_scraper import ArxivScraper
from document_processor import MemoryEfficientDocumentProcessor
from vector_db_interface import VectorDBInterface
from models import ResearchPlan, DocumentChunk, QualityMetricsReport, AnalysisResult, DraftContent, DraftSection, DraftMetadata
from utils import logger, time_it, extract_json, WorkflowState, safe_get
import json
import argparse
import yaml
from datetime import datetime
import nltk
# -----------------------------------------------------------------------------------------
# Research Planning Node
# -----------------------------------------------------------------------------------------

def research_planning_node(llm_interface: LLMInterface):
    def research_planning(state: WorkflowState) -> WorkflowState:
        """Node for research planning."""
        research_plan = safe_get(state, "research_plan")
        if not research_plan:
            logger.error("No research plan found in state")
            return state
            
        topic = research_plan.topic
        logger.info(f"Starting Research Planning for topic: {topic}")

        prompt = f"""
        As an expert research strategist, create a comprehensive research plan for: '{topic}'
        
        Provide a structured research plan with:
        1. Research Scope: Define clear boundaries and key aspects to investigate
        2. Search Terms: Generate 8-10 search terms, including:
           - Core topic terms
           - Related methodologies
           - Key theoretical frameworks
           - Relevant applications
        3. Exclusion Criteria: Specify what should be excluded from the review
        
        IMPORTANT: Respond with ONLY a valid JSON object in this format:
        {{
            "research_scope": "detailed scope description",
            "search_terms": ["term1", "term2", "term3"],
            "exclusion_criteria": ["criterion1", "criterion2"]
        }}
        """

        try:
            response_json_str = llm_interface.generate_text(prompt)
            logger.debug(f"LLM Response (Raw):\n{response_json_str}")

            # Clean up the response to extract just the JSON part
            cleaned_json_str = extract_json(response_json_str)
            if not cleaned_json_str.endswith("}"):
                cleaned_json_str += "}"  # Ensure JSON is properly closed
            
            try:
                response_json = json.loads(cleaned_json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error: {e}. Raw response: '{response_json_str}'")
                # Provide default values if parsing fails
                response_json = {
                    "research_scope": f"Default scope for {topic}",
                    "search_terms": [topic.lower()],
                    "exclusion_criteria": []
                }

            research_scope = response_json.get("research_scope", "No scope defined.")
            search_terms = response_json.get("search_terms", [])
            exclusion_criteria = response_json.get("exclusion_criteria", [])

            if not search_terms:  # Ensure we have at least one search term
                search_terms = [topic.lower()]

            # Update state with new research plan
            state["research_plan"] = ResearchPlan(
                topic=topic,
                search_terms=search_terms,
                timeline="To be defined",
                exclusion_criteria=exclusion_criteria
            )
            
            # Add planning result to messages
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Research plan created with {len(search_terms)} search terms")
            ]
            
            logger.info(f"Research Plan Created:\nScope: {research_scope}\nTerms: {search_terms}")

        except Exception as e:
            logger.error(f"Error in research planning: {e}")
            # Provide fallback behavior
            state["research_plan"] = ResearchPlan(
                topic=topic,
                search_terms=[topic.lower()],
                timeline="To be defined",
                exclusion_criteria=[]
            )
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Error in research planning: {str(e)}. Using default search terms.")
            ]

        return state

    return research_planning

# -----------------------------------------------------------------------------------------
# Data Collection Node
# -----------------------------------------------------------------------------------------

def data_collection_node(config: Config, db_interface: VectorDBInterface, 
                              document_processor: MemoryEfficientDocumentProcessor):
    def data_collection(state: Dict) -> Dict:
        """Node for data collection with improved error handling."""
        research_plan = state["research_plan"]
        logger.info(f"Starting Data Collection for: {research_plan.topic}")

        if not research_plan.search_terms:
            logger.warning("No search terms available.")
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="No search terms available for data collection.")
            ]
            return state

        # Create compound query with proper boolean operators
        search_terms = [f'"{term}"' for term in research_plan.search_terms]  # Quote terms
        combined_query = " OR ".join(search_terms)
        logger.info(f"Search Query: {combined_query}")

        try:
            # Process papers using existing document processor
            document_processor.process_arxiv_papers(combined_query)
            
            collection_size = db_interface.get_collection_size()
            
            message = f"Collected and processed documents. Total documents in collection: {collection_size}"
            logger.info(message)
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=message)
            ]

        except Exception as e:
            error_msg = f"Data collection error: {str(e)}"
            logger.error(error_msg)
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=error_msg)
            ]

        return state

    return data_collection

# -----------------------------------------------------------------------------------------
# Analysis Node
# -----------------------------------------------------------------------------------------

def analysis_node(state: Dict, db_interface: VectorDBInterface, llm_interface: LLMInterface) -> Dict:
    """Node for analysis of collected documents."""
    research_plan = state["research_plan"]
    topic = research_plan.topic
    search_terms = research_plan.search_terms
    logger.info(f"Starting Analysis Node for topic: '{topic}'")

    if not search_terms:
        logger.warning("No search terms available for analysis. Using topic as query.")
        query = topic
    else:
        query = " OR ".join(search_terms)
        logger.info(f"Analysis Query: '{query}'")

    relevant_chunks: List[DocumentChunk] = db_interface.search_similarity(query=query, top_k=10)
    if not relevant_chunks:
        logger.warning("No relevant document chunks found for analysis.")
        # Provide default analysis result
        default_result = AnalysisResult(
            key_themes=[topic],
            common_methodologies="No methodologies identified in the current dataset.",
            comparative_findings="Insufficient data for comparative analysis.",
            research_gaps=[f"Comprehensive research needed on {topic}"]
        )
        state["analysis_result"] = default_result
        return state

    logger.info(f"Retrieved {len(relevant_chunks)} relevant document chunks for analysis.")
    chunk_contents = "\n\n".join([f"--- Chunk from: {c.metadata.source_document_title} ---\n{c.content}" 
                                 for c in relevant_chunks])

    analysis_prompt = f"""
    As an expert research analyst, perform a detailed analysis of these document excerpts about '{topic}'.
    
    Focus on:
    1. Identify emerging patterns and relationships between key concepts
    2. Evaluate the strength and quality of evidence
    3. Compare and contrast different methodological approaches
    4. Highlight significant agreements and contradictions
    5. Identify gaps and opportunities for future research
    
    Document Excerpts:
    {chunk_contents}
    
    IMPORTANT: Respond ONLY with a valid JSON object in this format:
    {{
        "key_themes": [
            {{"theme": "theme name", "evidence_strength": "high", "description": "description text"}},
            {{"theme": "another theme", "evidence_strength": "medium", "description": "description text"}}
        ],
        "methodologies": [
            {{"method": "method name", "frequency": "common", "effectiveness": "effectiveness description"}},
            {{"method": "another method", "frequency": "rare", "effectiveness": "effectiveness description"}}
        ],
        "comparative_analysis": {{
            "agreements": ["agreement point 1", "agreement point 2"],
            "contradictions": ["contradiction point 1", "contradiction point 2"],
            "dependencies": ["dependency 1", "dependency 2"]
        }},
        "research_gaps": [
            {{"gap": "gap description", "importance": "high", "rationale": "explanation text"}},
            {{"gap": "another gap", "importance": "medium", "rationale": "explanation text"}}
        ]
    }}
    """

    try:
        analysis_json_str = llm_interface.generate_text(analysis_prompt)
        logger.debug(f"LLM Analysis Response (Raw):\n{analysis_json_str}")

        # Extract JSON from response
        cleaned_json_str = extract_json(analysis_json_str)
        try:
            analysis_json = json.loads(cleaned_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error in analysis: {e}. Raw response: '{analysis_json_str}'")
            # Provide structured default values
            analysis_json = {
                "key_themes": [
                    {"theme": topic, "evidence_strength": "medium", "description": "Primary research topic"},
                    {"theme": search_terms[0] if search_terms else "", "evidence_strength": "low", "description": "Related concept"}
                ],
                "methodologies": [
                    {"method": "Literature Review", "frequency": "common", "effectiveness": "Standard approach"}
                ],
                "comparative_analysis": {
                    "agreements": ["Further research needed"],
                    "contradictions": ["Varying approaches exist"],
                    "dependencies": ["Context-dependent findings"]
                },
                "research_gaps": [
                    {"gap": f"Comprehensive analysis of {topic}", "importance": "high", "rationale": "Need for systematic review"}
                ]
            }

        # Extract and format the analysis results
        key_themes = [item["theme"] for item in analysis_json.get("key_themes", [])]
        if not key_themes:
            key_themes = [topic]

        methodologies = [
            f"{item['method']} ({item['frequency']}, {item['effectiveness']})"
            for item in analysis_json.get("methodologies", [])
        ]
        if not methodologies:
            methodologies = ["Methodology analysis pending further review."]

        comparative = analysis_json.get("comparative_analysis", {})
        comparative_findings = (
            f"Agreements: {', '.join(comparative.get('agreements', []))}\n"
            f"Contradictions: {', '.join(comparative.get('contradictions', []))}\n"
            f"Context Dependencies: {', '.join(comparative.get('dependencies', []))}"
        )

        research_gaps = [
            f"{item['gap']} (Importance: {item['importance']})"
            for item in analysis_json.get("research_gaps", [])
        ]
        if not research_gaps:
            research_gaps = [f"Further research needed on {topic}"]

        # Create AnalysisResult object
        analysis_result = AnalysisResult(
            key_themes=key_themes,
            common_methodologies="\n".join(methodologies),
            comparative_findings=comparative_findings,
            research_gaps=research_gaps
        )
        
        state["analysis_result"] = analysis_result
        logger.info(f"Analysis Completed. Key Themes: {key_themes}")

    except Exception as e:
        logger.error(f"Error in analysis_node: {e}")
        # Provide fallback analysis result
        fallback_result = AnalysisResult(
            key_themes=[topic],
            common_methodologies="Analysis encountered technical difficulties.",
            comparative_findings="Further analysis needed.",
            research_gaps=[f"Systematic investigation needed for {topic}"]
        )
        state["analysis_result"] = fallback_result

    return state

# -----------------------------------------------------------------------------------------
# Draft Generation Node
# -----------------------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Count words in text, handling special characters and whitespace."""
    return len(text.split())

def draft_generation_node(llm_interface: LLMInterface) -> callable:
    """Creates a node for generating literature review draft sections."""
    def draft_generation(state: WorkflowState) -> WorkflowState:
        """Node for generating literature review draft sections."""
        research_plan = safe_get(state, "research_plan")
        analysis_result = safe_get(state, "analysis_result")
        
        if not analysis_result:
            logger.warning("No analysis results available for draft generation.")
            state["draft_content"] = DraftContent(
                formatted_content="Insufficient analysis results for draft generation."
            )
            return state

        logger.info(f"Starting Draft Generation for topic: {research_plan.topic}")

        # Construct the draft generation prompt
        prompt = f"""
        As an expert academic writer, create a comprehensive literature review for: '{research_plan.topic}'
        
        Use this analysis:
        Key Themes: {analysis_result.key_themes}
        Methodologies: {analysis_result.common_methodologies}
        Findings: {analysis_result.comparative_findings}
        Gaps: {analysis_result.research_gaps}
        
        Writing Requirements:
        1. Use formal academic language and proper citations
        2. Maintain logical flow between sections
        3. Provide critical analysis, not just description
        4. Support claims with evidence from the literature
        5. Highlight methodological strengths and limitations
        
        Structure each section with:
        - Clear topic sentences
        - Supporting evidence
        - Critical analysis
        - Transition sentences
        
        IMPORTANT: Format response as JSON:
        {{
            "sections": {{
                "introduction": {{
                    "content": "text",
                    "key_points": ["point1", "point2"]
                }},
                "methodology_review": {{
                    "content": "text",
                    "key_points": ["point1", "point2"]
                }},
                "findings_synthesis": {{
                    "content": "text",
                    "key_points": ["point1", "point2"]
                }},
                "critical_analysis": {{
                    "content": "text",
                    "key_points": ["point1", "point2"]
                }},
                "future_directions": {{
                    "content": "text",
                    "key_points": ["point1", "point2"]
                }}
            }},
            "metadata": {{
                "word_count": 0,
                "section_count": 5,
                "citation_count": 0
            }}
        }}
        """

        try:
            response_json_str = llm_interface.generate_text(prompt)
            logger.debug(f"LLM Draft Response (Raw):\n{response_json_str}")

            # Extract and parse JSON
            cleaned_json_str = extract_json(response_json_str)
            try:
                draft_json = json.loads(cleaned_json_str)
                
                # Get sections and ensure they're properly formatted
                sections = draft_json.get('sections', {})
                formatted_sections = {}
                
                # Extract content from each section, handling both string and dict formats
                for section_name, section_data in sections.items():
                    if isinstance(section_data, dict):
                        formatted_sections[section_name] = section_data.get('content', '')
                    elif isinstance(section_data, str):
                        formatted_sections[section_name] = section_data
                    else:
                        formatted_sections[section_name] = str(section_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error in draft generation: {e}")
                formatted_sections = {
                    "introduction": f"Introduction to {research_plan.topic}",
                    "methodology_review": "Methodology Review",
                    "findings_synthesis": "Findings Synthesis",
                    "critical_analysis": "Critical Analysis",
                    "future_directions": "Future Directions"
                }

            # Create DraftSection object with the formatted sections
            draft_section = DraftSection(
                overview=formatted_sections.get('introduction', ''),
                themes=formatted_sections.get('methodology_review', ''),
                methods=formatted_sections.get('findings_synthesis', ''),
                comparison=formatted_sections.get('critical_analysis', ''),
                gaps=formatted_sections.get('future_directions', '')
            )

            # Count words in each section
            total_words = sum(len(section.split()) for section in formatted_sections.values())

            # Create DraftMetadata object
            draft_metadata = DraftMetadata(
                word_count=total_words,
                section_count=len(formatted_sections),
                last_updated=datetime.now().isoformat(),
                version=1,
                citation_count=draft_json.get('metadata', {}).get('citation_count', 0)
            )

            # Format the draft in markdown with the correct section attributes
            formatted_content = f"""# Literature Review: {research_plan.topic}

## Overview
{draft_section.overview}

## Key Themes and Findings
{draft_section.themes}

## Methodological Approaches
{draft_section.methods}

## Comparative Analysis
{draft_section.comparison}

## Research Gaps and Future Directions
{draft_section.gaps}
"""
            # Create DraftContent object
            draft_content = DraftContent(
                draft_sections=draft_section,
                metadata=draft_metadata,
                formatted_content=formatted_content
            )

            # Store DraftContent object in state
            state["draft_content"] = draft_content
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Draft generated successfully with "
                         f"{draft_metadata.word_count} words across "
                         f"{draft_metadata.section_count} sections")
            ]
            
            logger.info(f"Draft generation completed successfully. Word count: {total_words}")

        except Exception as e:
            logger.error(f"Error in draft generation: {e}")
            # Update fallback section with correct attributes
            fallback_section = DraftSection(
                overview=f"This literature review examines {research_plan.topic}.",
                themes=', '.join(analysis_result.key_themes),
                methods=analysis_result.common_methodologies,
                comparison=analysis_result.comparative_findings,
                gaps=', '.join(analysis_result.research_gaps)
            )
            
            # Update fallback content to match section names
            fallback_content = f"""# Literature Review: {research_plan.topic}

## Overview
{fallback_section.overview}

## Key Themes and Findings
{fallback_section.themes}

## Methodological Approaches
{fallback_section.methods}

## Comparative Analysis
{fallback_section.comparison}

## Research Gaps and Future Directions
{fallback_section.gaps}
"""
            # Create DraftContent object
            state["draft_content"] = DraftContent(
                draft_sections=fallback_section,
                metadata=DraftMetadata(
                    word_count=len(fallback_content.split()),
                    section_count=5,
                    last_updated=datetime.now().isoformat(),
                    version=0,
                    citation_count=0
                ),
                formatted_content=fallback_content
            )

            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Error in draft generation: {str(e)}. Created fallback content.")
            ]

        return state

    return draft_generation

# -----------------------------------------------------------------------------------------
# Fact Checking Node - UPDATED NODE (Version 3 - LLM Verification)
# -----------------------------------------------------------------------------------------

def extract_json_from_llm_response(response: str) -> dict:
    """Extract JSON from LLM response, handling various response formats."""
    try:
        # First try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON-like structure in the response
        import re
        
        # More robust JSON pattern without recursive matching
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, response, re.DOTALL)
        
        # Try each potential JSON match
        for match in matches:
            try:
                json_str = match.group()
                # Clean up common formatting issues
                json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r':\s*"([^"]*)"(\s*[,}])', r':"\1"\2', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try extracting components with more flexible patterns
        status_pattern = r'(?:"?verification_status"?|status)\s*:\s*"?([^",}\s]+)"?'
        confidence_pattern = r'(?:"?confidence_score"?|confidence)\s*:\s*([\d.]+)'
        reason_pattern = r'(?:"?reason"?|explanation)\s*:\s*"([^"]+)"'
        
        status_match = re.search(status_pattern, response, re.IGNORECASE)
        confidence_match = re.search(confidence_pattern, response, re.IGNORECASE)
        reason_match = re.search(reason_pattern, response, re.IGNORECASE)
        
        if status_match or confidence_match or reason_match:
            return {
                "verification_status": status_match.group(1) if status_match else "needs review",
                "confidence_score": float(confidence_match.group(1)) if confidence_match else 0.5,
                "reason": reason_match.group(1) if reason_match else "No explicit reason provided"
            }
        
        # If all else fails, analyze the response text for verification indicators
        response_lower = response.lower()
        if any(word in response_lower for word in ["verified", "supported", "confirms"]):
            return {
                "verification_status": "verified",
                "confidence_score": 0.8,
                "reason": "Implicit verification found in response"
            }
        elif any(word in response_lower for word in ["contradicted", "conflicts", "disagrees"]):
            return {
                "verification_status": "contradicted",
                "confidence_score": 0.8,
                "reason": "Implicit contradiction found in response"
            }
        
        # Default fallback
        return {
            "verification_status": "needs review",
            "confidence_score": 0.5,
            "reason": f"Unable to parse verification details from response: {response[:100]}..."
        }

def fact_checking_node(db_interface: VectorDBInterface, llm_interface: LLMInterface):
    def fact_checking(state: WorkflowState) -> WorkflowState:
        """Node for fact-checking the draft content with improved verification."""
        draft_content: DraftContent = safe_get(state, "draft_content")
        if not draft_content or not draft_content.formatted_content:
            logger.warning("No draft content available for fact-checking.")
            return state

        logger.info("Starting Fact Checking Node (Version 3 - LLM Verification)...")
        markdown_draft = draft_content.formatted_content # Get Markdown draft

        # --- Claim Extraction (using NLTK sentence tokenizer) ---
        tokenizer = nltk.tokenize.sent_tokenize
        claims = tokenizer(markdown_draft) # Use NLTK sentence tokenizer
        logger.info(f"Extracted {len(claims)} claims for fact-checking (using NLTK).")

        # --- Source Retrieval and Verification (LLM-based Verification) ---
        verified_claims_count = 0
        needs_review_claims_count = 0
        unverified_claims_count = 0
        contradicted_claims_count = 0
        total_confidence_score = 0.0 # Track total confidence score

        verification_results = []

        annotated_draft = "" # Initialize empty annotated draft

        for i, claim in enumerate(claims):
            if not claim.strip(): # Skip empty claims
                annotated_draft += claim + "\n" # Keep empty lines
                continue

            logger.debug(f"Fact-checking claim {i+1}: '{claim[:100]}...'") # Log first 100 chars

            # --- Source Retrieval ---
            query = claim # Use claim as query for retrieval (improve query formulation later)
            relevant_chunks: List[DocumentChunk] = db_interface.search_similarity(query=query, top_k=3) # Retrieve top 3

            verification_status = "needs review" # Default status if verification fails
            confidence_score = 0.5 # Default confidence
            reason = "Initial assessment pending LLM verification." # Default reason

            if relevant_chunks:
                logger.debug(f"Retrieved {len(relevant_chunks)} chunks for claim {i+1}.")

                # --- LLM-Based Verification Prompt ---
                source_excerpts = "\n".join([f"--- Source {j+1} from: {chk.metadata.source_document_title} ---\n{chk.content}" for j, chk in enumerate(relevant_chunks)])
                verification_prompt = f"""
You are an expert fact-checker tasked with verifying claims against source documents.

CLAIM TO VERIFY:
"{claim}"

SOURCE DOCUMENT EXCERPTS:
{source_excerpts}

VERIFICATION RULES:
1. "verified" status requires:
   - Direct evidence from sources that explicitly supports the claim
   - Confidence > 0.8 only if evidence is clear and unambiguous
   - Must cite specific evidence in reason

2. "contradicted" status requires:
   - Direct evidence that explicitly conflicts with the claim
   - Must quote the contradicting evidence in reason
   - Assign confidence based on strength of contradiction

3. "needs review" status applies when:
   - Evidence is indirect or partially supporting
   - Sources are relevant but not conclusive
   - Insufficient evidence to verify or contradict
   - Multiple conflicting pieces of evidence

RESPONSE REQUIREMENTS:
1. Respond ONLY with a valid JSON object
2. Include specific quotes or evidence in reason
3. Confidence score must reflect evidence strength
4. Default to "needs review" if uncertain

EXAMPLE RESPONSES:

For Strong Evidence:
{{
    "verification_status": "verified",
    "confidence_score": 0.9,
    "reason": "Source directly states: '[exact quote supporting claim]'"
}}

For Contradictory Evidence:
{{
    "verification_status": "contradicted",
    "confidence_score": 0.85,
    "reason": "Source contradicts claim with: '[exact quote that conflicts]'"
}}

For Unclear Evidence:
{{
    "verification_status": "needs review",
    "confidence_score": 0.5,
    "reason": "Sources discuss [topic] but don't directly address [specific claim aspect]"
}}

YOUR RESPONSE (JSON only):
"""

                try:
                    llm_response_str = llm_interface.generate_text(verification_prompt)
                    logger.debug(f"LLM Verification Response (Raw):\n{llm_response_str}")
                    
                    # Use new extract_json_from_llm_response function
                    llm_response_json = extract_json_from_llm_response(llm_response_str)
                    
                    verification_status = safe_get(llm_response_json, "verification_status", "needs review")
                    confidence_score = float(safe_get(llm_response_json, "confidence_score", 0.5))
                    reason = safe_get(llm_response_json, "reason", "Verification assessment by LLM.")
                    
                    # Track verification statistics
                    total_confidence_score += confidence_score
                    
                    if verification_status == "verified":
                        verified_claims_count += 1
                    elif verification_status == "contradicted":
                        contradicted_claims_count += 1
                    else:  # "needs review" or any other status
                        needs_review_claims_count += 1
                        
                except Exception as e:
                    logger.error(f"Error during LLM-based verification: {e}")
                    verification_status = "needs review"
                    confidence_score = 0.5
                    reason = f"Error during verification: {str(e)}"
                    needs_review_claims_count += 1

            else: # No relevant chunks found
                logger.warning(f"No relevant sources found for claim {i+1}: '{claim[:100]}...'")
                verification_status = "unverified - needs review"
                unverified_claims_count += 1
                reason = "No relevant sources found for verification."

            # Create verification stats dictionary for each claim
            verification_stats = {
                "claim_index": i,
                "claim_text": claim,
                "verification_status": verification_status,
                "confidence_score": confidence_score,
                "reason": reason,
                "has_sources": bool(relevant_chunks),
                "source_count": len(relevant_chunks) if relevant_chunks else 0
            }

            # Track all verification results
            verification_results.append(verification_stats)

            # --- Annotation (Bold text annotation with status and reason in comment) ---
            annotation_prefix = f"**[{verification_status.title()}]**: " # Bold text annotation
            annotation_comment = f"<!-- Verification Status: {verification_status} | Confidence: {confidence_score:.2f} | Reason: {reason} -->" # Detailed comment
            annotated_claim = f"{annotation_prefix}{claim} {annotation_comment}" # Combine claim, annotation, and comment
            annotated_draft += annotated_claim + "\n\n" # Add claim, annotation, and comment, with double newline


        logger.info("Fact Checking Node (Version 3 - LLM Verification) completed.")
        # --- Update state ---
        draft_content.formatted_content = annotated_draft # Update draft with annotations

        quality_report: QualityMetricsReport = state.get("quality_report") or QualityMetricsReport()
        
        # Calculate verification statistics
        total_claims = len([c for c in claims if c.strip()])  # Count only non-empty claims
        if total_claims > 0:
            # Count verified claims with high confidence
            verified_high_confidence = len([
                v for v in verification_results 
                if v["verification_status"].lower() == "verified" 
                and v["confidence_score"] > 0.8
            ])
            
            # Calculate percentage based on verified high-confidence claims
            quality_report.claim_verification_percentage = (verified_high_confidence / total_claims * 100)
            
            # Add detailed verification metrics
            quality_report.metrics_passed.update({
                "fact_checking_verified_claims": verified_claims_count > 0,
                "fact_checking_high_confidence": verified_high_confidence / total_claims > 0.5,
                "has_contradictions": contradicted_claims_count > 0,
                "needs_review": needs_review_claims_count > 0
            })
            
            # Add comprehensive verification report with more detail
            quality_report.hallucination_report = [
                f"Verification Summary:",
                f"- Total Claims: {total_claims}",
                f"- Verified (High Confidence): {verified_high_confidence} ({quality_report.claim_verification_percentage:.1f}%)",
                f"- Verified (Any Confidence): {verified_claims_count} ({(verified_claims_count/total_claims*100):.1f}%)",
                f"- Contradicted: {contradicted_claims_count} ({(contradicted_claims_count/total_claims*100):.1f}%)",
                f"- Needs Review: {needs_review_claims_count} ({(needs_review_claims_count/total_claims*100):.1f}%)",
                f"- Average Confidence: {total_confidence_score/total_claims:.2f}",
                "\nDetailed Verification Results:",
                *[f"Claim {v['claim_index']+1}: {v['verification_status'].title()} "
                  f"(Confidence: {v['confidence_score']:.2f})"
                  for v in verification_results]
            ]

        state["quality_report"] = quality_report

        return state

    return fact_checking

# -----------------------------------------------------------------------------------------
# Quality Check Node
# -----------------------------------------------------------------------------------------

def quality_check_node(llm_interface: LLMInterface):
    def quality_check(state: WorkflowState) -> WorkflowState:
        """Node for checking the quality of the draft before fact checking."""
        draft_content = safe_get(state, "draft_content")
        if not draft_content:
            return state
            
        prompt = f"""
        As an academic writing expert, evaluate this draft for quality:
        
        {draft_content.formatted_content}
        
        Check for:
        1. Academic language and tone
        2. Logical flow and coherence
        3. Depth of analysis
        4. Evidence support
        5. Citation usage
        
        IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
        {{
            "quality_scores": {{
                "academic_tone": 7,
                "coherence": 8,
                "analysis_depth": 6,
                "evidence_support": 7,
                "citation_usage": 5
            }},
            "improvement_suggestions": [
                {{"section": "Introduction", "issue": "issue description", "suggestion": "improvement suggestion"}},
                {{"section": "Methods", "issue": "issue description", "suggestion": "improvement suggestion"}}
            ],
            "requires_revision": true
        }}
        """
        
        try:
            response = llm_interface.generate_text(prompt)
            cleaned_json = extract_json(response)
            quality_check_result = json.loads(cleaned_json)
            
            # Validate the quality check result
            if not isinstance(quality_check_result, dict):
                raise ValueError("Quality check result must be a dictionary")
                
            # Ensure required fields exist
            quality_check_result.setdefault("quality_scores", {
                "academic_tone": 5,
                "coherence": 5,
                "analysis_depth": 5,
                "evidence_support": 5,
                "citation_usage": 5
            })
            quality_check_result.setdefault("improvement_suggestions", [])
            quality_check_result.setdefault("requires_revision", True)
            
            # Update state with quality check results
            state["quality_check"] = quality_check_result
            
            # Add quality check message
            message = (
                f"Quality Check Results:\n"
                f"Academic Tone: {quality_check_result['quality_scores']['academic_tone']}/10\n"
                f"Coherence: {quality_check_result['quality_scores']['coherence']}/10\n"
                f"Analysis Depth: {quality_check_result['quality_scores']['analysis_depth']}/10\n"
                f"Evidence Support: {quality_check_result['quality_scores']['evidence_support']}/10\n"
                f"Citation Usage: {quality_check_result['quality_scores']['citation_usage']}/10\n"
                f"\nRequires Revision: {quality_check_result['requires_revision']}"
            )
            
            state["messages"].append(AIMessage(content=message))
                
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
            # Provide default quality check result
            state["quality_check"] = {
                "quality_scores": {
                    "academic_tone": 5,
                    "coherence": 5,
                    "analysis_depth": 5,
                    "evidence_support": 5,
                    "citation_usage": 5
                },
                "improvement_suggestions": [
                    {"section": "General", "issue": "Quality check failed", "suggestion": "Please review manually"}
                ],
                "requires_revision": True
            }
            state["messages"].append(
                AIMessage(content=f"Error in quality check: {str(e)}. Using default values.")
            )
            
        return state
        
    return quality_check

# -----------------------------------------------------------------------------------------
# Workflow Creation
# -----------------------------------------------------------------------------------------

@time_it
def create_research_agent_workflow(config: Config, llm_interface: LLMInterface,
                                 db_interface: VectorDBInterface,
                                 document_processor: MemoryEfficientDocumentProcessor) -> StateGraph:
    """Creates the LangGraph workflow including Fact Checking Node (Version 3)."""
    logger.info("Creating Research Agent LangGraph Workflow (with Fact Checking Node - Version 3)...")

    builder = StateGraph(Dict)

    # Add state validation
    def validate_state(state: Dict) -> Dict:
        """Validates and initializes state with required fields."""
        required_fields = {
            "research_plan": ResearchPlan(topic=""),
            "document_chunks": [],
            "quality_report": QualityMetricsReport(),
            "analysis_result": None,
            "draft_content": None,
            "draft_metadata": {},
            "messages": [],
            "processed_documents": []
        }
        
        for field, default in required_fields.items():
            if field not in state:
                state[field] = default
        return state

    # Add nodes with validation
    builder.add_node("state_validation", validate_state)
    builder.add_node("research_planning", research_planning_node(llm_interface))
    builder.add_node("data_collection", data_collection_node(config, db_interface, document_processor))
    builder.add_node("analysis", lambda x: analysis_node(x, db_interface, llm_interface))
    builder.add_node("draft_generation", draft_generation_node(llm_interface))
    builder.add_node("quality_check", quality_check_node(llm_interface))
    builder.add_node("fact_checking", fact_checking_node(db_interface, llm_interface)) # Pass llm_interface to fact_checking_node

    # Add edges with validation
    builder.add_edge("state_validation", "research_planning")
    builder.add_edge("research_planning", "data_collection")
    builder.add_edge("data_collection", "analysis")
    builder.add_edge("analysis", "draft_generation")
    builder.add_edge("draft_generation", "quality_check")
    builder.add_edge("quality_check", "fact_checking")
    builder.add_edge("fact_checking", END)

    builder.set_entry_point("state_validation")

    try:
        graph = builder.compile()
        logger.info("Research Agent LangGraph Workflow (with Fact Checking Node - Version 3) created.")
        return graph
    except Exception as e:
        logger.error(f"Error building workflow graph: {e}")
        raise

# -----------------------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------------------

@time_it
def run_research_agent(topic: str, model_name: str = None):
    """Main function to run the research agent workflow (prints Fact-Checked Draft)"""
    logger.info(f"Starting Research Agent for topic: '{topic}'")

    config = load_config()
    
    # Override model name if provided
    if model_name:
        config.llm.model_name = model_name
        logger.info(f"Using override model: {model_name}")

    llm_interface = LLMInterface(config)
    llm_interface.initialize_llm()
    db_interface = VectorDBInterface(config)
    db_interface.initialize_db()
    document_processor = MemoryEfficientDocumentProcessor(config, db_interface)

    workflow = create_research_agent_workflow(config, llm_interface, db_interface, document_processor)

    initial_state = {
        "research_plan": ResearchPlan(topic=topic),
        "document_chunks": [],
        "quality_report": QualityMetricsReport(),
        "analysis_result": None,
        "draft_metadata": {},
        "messages": []
    }

    try:
        logger.info("Executing LangGraph workflow (with Fact Checking Node)...")
        results = workflow.invoke(initial_state)
        logger.info("LangGraph workflow execution completed (with Fact Checking Node).")

        if results.get("research_plan"):
            logger.info("\n--- Final Research Plan ---")
            logger.info(f"Topic: {results['research_plan'].topic}")
            logger.info(f"Search Terms: {results['research_plan'].search_terms}")

        if results.get("analysis_result"):
            logger.info("\n--- Analysis Results ---")
            logger.info(f"Key Themes: {results['analysis_result'].key_themes}")
            logger.info(f"Common Methodologies: {results['analysis_result'].common_methodologies}")
            logger.info(f"Comparative Findings: {results['analysis_result'].comparative_findings}")
            logger.info(f"Research Gaps: {results['analysis_result'].research_gaps}")

        if results.get("draft_content"):
            logger.info("\n--- Fact-Checked Draft ---") # Updated section title
            print(results["draft_content"].formatted_content) # Print annotated draft
            logger.info(f"Draft Statistics: Word Count: {results['draft_content'].metadata.word_count}")
        else:
            logger.warning("Draft content not found in final state.")

        if results.get("quality_report"): # Print quality metrics
            logger.info("\n--- Quality Metrics ---")
            logger.info(f"Claim Verification Percentage: {results['quality_report'].claim_verification_percentage:.2f}%")


        logger.info(f"Total Document Chunks in VectorDB: {db_interface.get_collection_size()}")
        logger.info(f"Total Words in Generated Draft: {results.get('draft_content').metadata.word_count if results.get('draft_content') else 0}")

    except Exception as e:
        logger.error(f"Error running LangGraph workflow (with Fact Checking Node): {e}")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run Research Agent with configurable LLM model')
    parser.add_argument('--topic', type=str, default="Agentic AI Systems in Accounting",
                       help='Research topic to analyze')
    parser.add_argument('--model', type=str, 
                       help='LLM model name (e.g., llama3.2:latest, mistral:latest, deepseek-r1:14b)')
    
    args = parser.parse_args()
    run_research_agent(args.topic, args.model)