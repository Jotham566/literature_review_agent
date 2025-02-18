# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ResearchPlan(BaseModel):
    topic: str
    search_terms: List[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)

class DocumentMetadata(BaseModel):
    source_document_title: str = ""  # Changed to non-Optional with empty string default
    source_document_type: str = "unknown"  # Changed to non-Optional with default
    source_url: str = ""  # Changed to non-Optional with empty string default
    source_file_path: str = ""  # Changed to non-Optional with empty string default
    publication_date: str = ""  # Changed to non-Optional with empty string default
    authors: List[str] = Field(default_factory=list)
    citation_string: str = ""  # Changed to non-Optional with empty string default
    page_number: str = ""  # Changed to non-Optional with empty string default
    chunk_id: str = ""  # Changed to non-Optional with empty string default
    source_document_id: str = ""

    def clean_for_chroma(self) -> dict:
        """Convert metadata to ChromaDB-compatible format."""
        clean_dict = self.model_dump()
        
        # Convert authors list to string
        clean_dict['authors'] = ', '.join(self.authors) if self.authors else ""
        
        # Ensure all values are strings, numbers, or booleans
        for key, value in clean_dict.items():
            if value is None:
                clean_dict[key] = ""
            elif not isinstance(value, (str, int, float, bool)):
                clean_dict[key] = str(value)
                
        return clean_dict

class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata

class Citation(BaseModel):
    authors: List[str] = Field(default_factory=list)
    title: str
    journal: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    accessed_date: Optional[str] = None
    verification_status: str = "unverified"  # Changed default from Optional
    completeness_score: float = 0.0

class QualityMetricsReport(BaseModel):
    source_diversity: int = 0
    citation_density: float = 0.0
    claim_verification_percentage: float = 0.0
    cross_reference_coverage: float = 0.0
    hallucination_report: List[str] = Field(default_factory=list)
    overall_quality_score: float = 0.0
    metrics_passed: Dict[str, bool] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    verification_metrics: Dict[str, Any] = Field(default_factory=dict)
    verification_report: Dict[str, str] = Field(default_factory=dict)

class AnalysisResult(BaseModel):
    key_themes: List[str] = Field(default_factory=list)
    common_methodologies: str = "No methodologies identified."
    comparative_findings: str = "No comparative findings."
    research_gaps: List[str] = Field(default_factory=list)

class DraftSection(BaseModel):
    overview: str = ""
    themes: str = ""
    methods: str = ""
    comparison: str = ""
    gaps: str = ""

class DraftMetadata(BaseModel):
    word_count: int = 0
    section_count: int = 5
    last_updated: Optional[str] = None
    version: int = 1

class DraftContent(BaseModel):
    draft_sections: DraftSection = Field(default_factory=DraftSection)
    metadata: DraftMetadata = Field(default_factory=DraftMetadata)
    formatted_content: str = ""  # The final markdown formatted content