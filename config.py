import yaml
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class LLMConfig(BaseModel):
    model_name: str = "mistral"
    fallback_model_name: str = "llama3.2:latest"
    params: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "max_tokens": 2000
    }

class VectorDBConfig(BaseModel):
    db_type: str = "chroma"
    persist_directory: str = "vector_db"
    chunk_size: int = 2000  # Increased for memory efficiency
    chunk_overlap: int = 200
    batch_size: int = 5
    retry_delay: int = 1
    max_retries: int = 3

class ProcessingConfig(BaseModel):
    paper_batch_size: int = 2
    max_workers: int = 4
    max_papers: int = 10
    force_gc: bool = True
    max_memory_percent: float = 75.0

class LogRotationConfig(BaseModel):
    max_size_mb: int = 10
    backup_count: int = 5

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_path: str = "logs/document_processor.log"
    console_output: bool = True
    detailed_memory_tracking: bool = True
    log_rotation: LogRotationConfig = LogRotationConfig()

class SourceCriteriaConfig(BaseModel):
    peer_reviewed: float = 2.0
    citation_count: float = 1.5
    publication_date: float = 1.0
    journal_impact: float = 1.5
    author_h_index: float = 1.0

class QualityThresholdsConfig(BaseModel):
    min_sources: int = 15
    min_citation_density: float = 0.8
    min_verification_score: float = 0.9
    max_uncertainty_level: float = 0.2
    min_source_credibility_score: float = 2.5

class CitationTemplateConfig(BaseModel):
    fields: List[str] = ["authors", "title", "journal", "year", "doi", "url", "accessed_date", "verification_status"]

class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    processing: ProcessingConfig = ProcessingConfig()
    logging: LoggingConfig = LoggingConfig()
    source_criteria: SourceCriteriaConfig = SourceCriteriaConfig()
    quality_thresholds: QualityThresholdsConfig = QualityThresholdsConfig()
    citation_template: CitationTemplateConfig = CitationTemplateConfig()

def load_config(config_path="config.yaml") -> Config:
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            if config_dict:
                return Config(**config_dict)
            else:
                return Config()  # Return default config if file is empty
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default configuration.")
        return Config()
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}. Using default configuration.")
        return Config()

if __name__ == "__main__":
    default_config = Config()
    print("Default Config:", default_config.model_dump())