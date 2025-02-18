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
        You are an expert research strategist. Your goal is to create a research plan for the topic: '{topic}'.
        
        Generate a research plan with the following components:
        1. Define Research Scope: Briefly outline the key aspects and boundaries of this literature review.
        2. Identify Key Search Terms: Generate 5-7 highly relevant search terms for academic databases.

        IMPORTANT: Respond ONLY with a valid JSON object in the following format, with no additional text before or after:
        {{
          "research_scope": "This review will focus on...",
          "search_terms": ["term1", "term2", ...]
        }}
        """

        try:
            response_json_str = llm_interface.generate_text(prompt)
            logger.debug(f"LLM Response (Raw):\n{response_json_str}")

            # Clean up the response to extract just the JSON part
            cleaned_json_str = extract_json(response_json_str)
            try:
                response_json = json.loads(cleaned_json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error: {e}. Raw response: '{response_json_str}'")
                # Provide default values if parsing fails
                response_json = {
                    "research_scope": "Default scope for " + topic,
                    "search_terms": [topic.lower()]
                }

            research_scope = response_json.get("research_scope", "No scope defined.")
            search_terms = response_json.get("search_terms", [])

            if not search_terms:  # Ensure we have at least one search term
                search_terms = [topic.lower()]

            # Update state with new research plan
            state["research_plan"] = ResearchPlan(
                topic=topic,
                search_terms=search_terms,
                timeline="To be defined"
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
                timeline="To be defined"
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
    """
    Node for analysis of collected documents.
    """
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
    You are an expert research analyst. Analyze these document excerpts about: '{topic}'.

    Document Excerpts:
    {chunk_contents}

    IMPORTANT: Respond ONLY with a valid JSON object in exactly this format, with no additional text:
    {{
        "key_themes": ["theme1", "theme2", "theme3"],
        "common_methodologies": "Description of methodologies found",
        "comparative_findings": "Summary of agreements and disagreements",
        "research_gaps": ["gap1", "gap2"]
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
            # Provide structured default values based on chunks
            analysis_json = {
                "key_themes": [topic] + [term for term in search_terms[:2]],
                "common_methodologies": "Analysis of methodologies pending structured review.",
                "comparative_findings": "Further comparative analysis needed.",
                "research_gaps": [
                    f"Systematic review needed for {topic}",
                    "Integration of multiple research perspectives required"
                ]
            }

        # Ensure all required fields exist with valid data
        key_themes = analysis_json.get("key_themes", [])
        if not key_themes:
            key_themes = [topic]

        common_methodologies = analysis_json.get("common_methodologies", "")
        if not common_methodologies.strip():
            common_methodologies = "Methodology analysis pending further review."

        comparative_findings = analysis_json.get("comparative_findings", "")
        if not comparative_findings.strip():
            comparative_findings = "Comparative analysis pending additional data."

        research_gaps = analysis_json.get("research_gaps", [])
        if not research_gaps:
            research_gaps = [f"Further research needed on {topic}"]

        analysis_result_obj = AnalysisResult(
            key_themes=key_themes,
            common_methodologies=common_methodologies,
            comparative_findings=comparative_findings,
            research_gaps=research_gaps
        )
        state["analysis_result"] = analysis_result_obj
        logger.info(f"Analysis Completed. Key Themes: {key_themes}, Research Gaps: {research_gaps}")

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
        You are an expert academic writer. Generate a literature review section for the topic: '{research_plan.topic}'.
        Use the following analysis results to create a well-structured, academic draft:

        Key Themes: {analysis_result.key_themes}
        Common Methodologies: {analysis_result.common_methodologies}
        Comparative Findings: {analysis_result.comparative_findings}
        Research Gaps: {analysis_result.research_gaps}

        Generate a draft with the following sections. For each section, provide a clear, detailed paragraph of text.
        1. Overview of the Field
        2. Key Themes and Findings
        3. Methodological Approaches
        4. Comparative Analysis
        5. Research Gaps and Future Directions

        IMPORTANT: Format your response as a valid JSON object with this exact structure, where each section value is a string:
        {{
            "sections": {{
                "overview": "text content as a single string",
                "themes": "text content as a single string",
                "methods": "text content as a single string",
                "comparison": "text content as a single string",
                "gaps": "text content as a single string"
            }},
            "metadata": {{
                "word_count": 0,
                "section_count": 5
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
                
                # Ensure all section values are strings
                sections = draft_json.get('sections', {})
                for key, value in sections.items():
                    if isinstance(value, dict):
                        # If value is a dict, convert it to a string
                        sections[key] = str(value)
                    elif not isinstance(value, str):
                        # If value is neither string nor dict, convert to string
                        sections[key] = str(value)
                
                draft_json['sections'] = sections
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error in draft generation: {e}")
                draft_json = {
                    "sections": {
                        "overview": f"Literature Review: {research_plan.topic}",
                        "themes": "Key themes analysis pending.",
                        "methods": "Methodological analysis pending.",
                        "comparison": "Comparative analysis pending.",
                        "gaps": "Research gaps analysis pending."
                    },
                    "metadata": {
                        "word_count": 0,
                        "section_count": 5
                    }
                }

            # Create DraftSection object
            draft_section = DraftSection(
                overview=str(draft_json['sections']['overview']),
                themes=str(draft_json['sections']['themes']),
                methods=str(draft_json['sections']['methods']),
                comparison=str(draft_json['sections']['comparison']),
                gaps=str(draft_json['sections']['gaps'])
            )

            # Count words in each section
            total_words = sum(count_words(section) 
                            for section in draft_json['sections'].values())

            # Create DraftMetadata object
            draft_metadata = DraftMetadata(
                word_count=total_words,
                section_count=len(draft_json['sections']),
                last_updated=datetime.now().isoformat(),
                version=1
            )

            # Format the draft in markdown
            formatted_content = f"""# Literature Review: {research_plan.topic}

## Overview of the Field
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

            # Store single DraftContent object in state
            state["draft_content"] = draft_content
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Draft generated successfully with "
                         f"{draft_metadata.word_count} words across "
                         f"{draft_metadata.section_count} sections")
            ]
            
            logger.info(f"Draft generation completed successfully. Word count: {total_words}")

        except Exception as e:
            logger.error(f"Error in draft generation: {e}")
            # Create fallback DraftContent object
            fallback_content = f"""# Literature Review: {research_plan.topic}

Draft generation encountered technical difficulties. Please retry the process.
Key themes identified: {', '.join(analysis_result.key_themes)}
"""
            fallback_section = DraftSection(
                overview="Draft generation failed",
                themes=f"Identified themes: {', '.join(analysis_result.key_themes)}",
                methods="Generation error",
                comparison="Generation error",
                gaps="Generation error"
            )
            
            fallback_metadata = DraftMetadata(
                word_count=len(fallback_content.split()),
                section_count=0,
                last_updated=datetime.now().isoformat(),
                version=0
            )

            state["draft_content"] = DraftContent(
                draft_sections=fallback_section,
                metadata=fallback_metadata,
                formatted_content=fallback_content
            )

            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Error in draft generation: {str(e)}")
            ]

        return state

    return draft_generation

# -----------------------------------------------------------------------------------------
# Fact Checking Node - UPDATED NODE (Version 2)
# -----------------------------------------------------------------------------------------

def fact_checking_node(db_interface: VectorDBInterface): # Pass db_interface
    def fact_checking(state: WorkflowState) -> WorkflowState:
        """Node for fact-checking the draft content (Version 2 - NLTK, improved placeholder status)."""
        draft_content: DraftContent = safe_get(state, "draft_content")
        if not draft_content or not draft_content.formatted_content:
            logger.warning("No draft content available for fact-checking.")
            return state

        logger.info("Starting Fact Checking Node (Version 2 - NLTK, improved placeholder status)...")
        markdown_draft = draft_content.formatted_content # Get Markdown draft

        # --- Claim Extraction (using NLTK sentence tokenizer) ---
        tokenizer = nltk.tokenize.sent_tokenize
        claims = tokenizer(markdown_draft) # Use NLTK sentence tokenizer
        logger.info(f"Extracted {len(claims)} claims for fact-checking (using NLTK).")

        # --- Source Retrieval and Verification (Placeholder - Improved status) ---
        verified_claims_count = 0
        needs_review_claims_count = 0
        unverified_claims_count = 0 # Added unverified count
        contradicted_claims_count = 0 # (Not used in this version, but can be added later)

        annotated_draft = "" # Initialize empty annotated draft

        for i, claim in enumerate(claims):
            if not claim.strip(): # Skip empty claims
                annotated_draft += claim + "\n" # Keep empty lines
                continue

            logger.debug(f"Fact-checking claim {i+1}: '{claim[:100]}...'") # Log first 100 chars

            # --- Source Retrieval ---
            query = claim # Use claim as query for retrieval (improve query formulation later)
            relevant_chunks: List[DocumentChunk] = db_interface.search_similarity(query=query, top_k=3) # Retrieve top 3

            if relevant_chunks:
                logger.debug(f"Retrieved {len(relevant_chunks)} chunks for claim {i+1}.")
                verification_status = "potentially supported - needs review" # Improved placeholder status
                needs_review_claims_count += 1 # Still counts as needs review for now
            else:
                logger.warning(f"No relevant sources found for claim {i+1}: '{claim[:100]}...'")
                verification_status = "unverified - needs review" # More informative unverified status
                unverified_claims_count += 1 # Count as unverified

            # --- Annotation (Bold text annotation) ---
            annotation_prefix = f"**[{verification_status.title()}]**: " # Bold text annotation
            annotated_claim = f"{annotation_prefix}{claim}"
            annotated_draft += annotated_claim + "\n\n" # Add claim and annotation, with double newline


        logger.info("Fact Checking Node (Version 2) completed.")
        # --- Update state ---
        draft_content.formatted_content = annotated_draft # Update draft with annotations

        quality_report: QualityMetricsReport = state.get("quality_report") or QualityMetricsReport() # Get or create QualityMetricsReport
        total_claims = len(claims) - annotated_draft.count("<!--") # Exclude comment lines from claim count (if any comments added later)
        if total_claims > 0:
            quality_report.claim_verification_percentage = (verified_claims_count / total_claims * 100) # Example metric (still based on 'verified' count which is 0 now)
        else:
            quality_report.claim_verification_percentage = 0.0

        state["quality_report"] = quality_report # Update quality report in state

        return state

    return fact_checking

# -----------------------------------------------------------------------------------------
# Workflow Creation
# -----------------------------------------------------------------------------------------

@time_it
def create_research_agent_workflow(config: Config, llm_interface: LLMInterface, 
                                 db_interface: VectorDBInterface, 
                                 document_processor: MemoryEfficientDocumentProcessor) -> StateGraph:
    """Creates the LangGraph workflow including Fact Checking Node."""
    logger.info("Creating Research Agent LangGraph Workflow (with Fact Checking Node)...")
    
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
    builder.add_node("fact_checking", fact_checking_node(db_interface))

    # Add edges with validation
    builder.add_edge("state_validation", "research_planning")
    builder.add_edge("research_planning", "data_collection")
    builder.add_edge("data_collection", "analysis")
    builder.add_edge("analysis", "draft_generation")
    builder.add_edge("draft_generation", "fact_checking")
    builder.add_edge("fact_checking", END)

    builder.set_entry_point("state_validation")

    try:
        graph = builder.compile()
        logger.info("Research Agent LangGraph Workflow (with Fact Checking Node) created.")
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