# langgraph_workflow.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Dict, Any, Tuple, Optional
from config import Config, load_config
from llm_interface import LLMInterface
from arxiv_scraper import ArxivScraper
from document_processor import MemoryEfficientDocumentProcessor
from vector_db_interface import VectorDBInterface
from models import ResearchPlan, DocumentChunk, QualityMetricsReport, AnalysisResult
from utils import logger, time_it
import json
import argparse
import yaml

# -----------------------------------------------------------------------------------------
# Research Planning Node
# -----------------------------------------------------------------------------------------

def research_planning_node(llm_interface: LLMInterface):
    def research_planning(state: Dict) -> Dict:
        """Node for research planning."""
        research_plan = state["research_plan"]
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
            def extract_json(text):
                # Find the first '{' and last '}'
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    return text[start:end + 1]
                return text

            # Clean and parse JSON
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
        def extract_json(text):
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return text[start:end + 1]
            return text

        # Clean and parse JSON
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
    def draft_generation(state: Dict) -> Dict:
        """Node for generating literature review draft sections."""
        research_plan = state["research_plan"]
        analysis_result = state.get("analysis_result")
        
        if not analysis_result:
            logger.warning("No analysis results available for draft generation.")
            state["draft_content"] = "Insufficient analysis results for draft generation."
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

        Generate a draft in markdown format with the following structure:
        1. Overview of the Field
        2. Key Themes and Findings
        3. Methodological Approaches
        4. Comparative Analysis
        5. Research Gaps and Future Directions

        IMPORTANT: Format your response as a valid JSON object with this exact structure:
        {{
            "sections": {{
                "overview": "text here",
                "themes": "text here",
                "methods": "text here",
                "comparison": "text here",
                "gaps": "text here"
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
            def extract_json(text):
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    return text[start:end]
                return text

            cleaned_json_str = extract_json(response_json_str)
            try:
                draft_json = json.loads(cleaned_json_str)
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

            # Count words in each section
            total_words = sum(count_words(section) 
                            for section in draft_json['sections'].values())
            
            # Update metadata with actual word count
            draft_json['metadata']['word_count'] = total_words

            # Format the draft in markdown
            draft_content = f"""# Literature Review: {research_plan.topic}

## Overview of the Field
{draft_json['sections']['overview']}

## Key Themes and Findings
{draft_json['sections']['themes']}

## Methodological Approaches
{draft_json['sections']['methods']}

## Comparative Analysis
{draft_json['sections']['comparison']}

## Research Gaps and Future Directions
{draft_json['sections']['gaps']}
"""
            state["draft_content"] = draft_content
            state["draft_metadata"] = {
                "word_count": total_words,
                "section_count": len(draft_json['sections'])
            }
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Draft generated successfully with "
                         f"{total_words} words across "
                         f"{len(draft_json['sections'])} sections")
            ]
            
            logger.info(f"Draft generation completed successfully. Word count: {total_words}")

        except Exception as e:
            logger.error(f"Error in draft generation: {e}")
            state["draft_content"] = f"""# Literature Review: {research_plan.topic}

Draft generation encountered technical difficulties. Please retry the process.
Key themes identified: {', '.join(analysis_result.key_themes)}
"""
            state["draft_metadata"] = {"word_count": 0, "section_count": 0}
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Error in draft generation: {str(e)}")
            ]

        return state

    return draft_generation

# -----------------------------------------------------------------------------------------
# Workflow Creation
# -----------------------------------------------------------------------------------------

@time_it
def create_research_agent_workflow(config: Config, llm_interface: LLMInterface, 
                                 db_interface: VectorDBInterface, 
                                 document_processor: MemoryEfficientDocumentProcessor) -> StateGraph:
    """Creates the LangGraph workflow with improved error handling and state management."""
    logger.info("Creating Research Agent LangGraph Workflow...")
    
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

    # Add edges with validation
    builder.add_edge("state_validation", "research_planning")
    builder.add_edge("research_planning", "data_collection")
    builder.add_edge("data_collection", "analysis")
    builder.add_edge("analysis", "draft_generation")
    builder.add_edge("draft_generation", END)

    builder.set_entry_point("state_validation")

    try:
        graph = builder.compile()
        logger.info("Research Agent LangGraph Workflow created successfully.")
        return graph
    except Exception as e:
        logger.error(f"Error building workflow graph: {e}")
        raise

# -----------------------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------------------

@time_it
def run_research_agent(topic: str, model_name: str = None):
    """Main function to run the research agent workflow."""
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
        "messages": []
    }

    try:
        logger.info("Executing LangGraph workflow...")
        results = workflow.invoke(initial_state)
        logger.info("LangGraph workflow execution completed.")

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
            logger.info("\n--- Generated Draft ---")
            print(results["draft_content"])
            if results.get("draft_metadata"):
                logger.info(f"Draft Statistics: {results['draft_metadata']}")
        else:
            logger.warning("Draft content not found in final state.")

        logger.info(f"Total Document Chunks in VectorDB: {db_interface.get_collection_size()}")

    except Exception as e:
        logger.error(f"Error running LangGraph workflow: {e}")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run Research Agent with configurable LLM model')
    parser.add_argument('--topic', type=str, default="Agentic AI Systems in Accounting",
                       help='Research topic to analyze')
    parser.add_argument('--model', type=str, 
                       help='LLM model name (e.g., llama3.2:latest, mistral:latest, deepseek-r1:14b)')
    
    args = parser.parse_args()
    run_research_agent(args.topic, args.model)