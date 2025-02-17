# document_processor.py 
'''
This script processes papers from ArXiv, extracts text from PDFs, 
and adds the text to a SQLite database.
'''
from typing import List, Generator, Optional
import pypdf
from tqdm import tqdm
import logging
import logging.handlers
import psutil
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from dataclasses import dataclass
from datetime import datetime
from arxiv_scraper import ArxivScraper
from vector_db_interface import VectorDBInterface
from config import Config, load_config
from models import DocumentChunk, DocumentMetadata
import numpy as np
import traceback
import argparse
from vector_db_interface import VectorDBInterface

# Suppress numpy scientific notation
np.set_printoptions(suppress=True)

@dataclass
class ProcessingStats:
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    peak_memory_usage: float = 0.0
    start_time: datetime = None
    end_time: datetime = None

class VerboseFilter(logging.Filter):
    def filter(self, record):
        return not any(x in str(record.getMessage()).lower() for x in [
            'dtype=float32', 
            'array([', 
            'shape=',
            'adding embedding',
            'dimension'
        ])

class ConfiguredLogger:
    @staticmethod
    def setup(config: Config):
        log_dir = Path(os.path.dirname(config.logging.file_path))
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger()
        logger.setLevel(getattr(logging, config.logging.level))
        logger.handlers.clear()

        verbose_filter = VerboseFilter()
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.log_rotation.max_size_mb * 1024 * 1024,
            backupCount=config.logging.log_rotation.backup_count
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        file_handler.addFilter(verbose_filter)
        logger.addHandler(file_handler)

        if config.logging.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            console_handler.addFilter(verbose_filter)
            logger.addHandler(console_handler)

        # Suppress other verbose loggers
        for log_name in ['chromadb', 'sentence_transformers']:
            other_logger = logging.getLogger(log_name)
            other_logger.setLevel(logging.WARNING)
            other_logger.addFilter(verbose_filter)
        
        return logger

class MemoryEfficientDocumentProcessor:
    def __init__(self, config: Config, db_interface: VectorDBInterface):
        self.config = config
        self.db_interface = db_interface
        self.arxiv_scraper = ArxivScraper()
        self.stats = ProcessingStats()
        self.logger = ConfiguredLogger.setup(config)
        self.stats.start_time = datetime.now()
        
        # Create output directories
        Path("arxiv_papers").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def _check_duplicate(self, paper) -> bool:
        """Check if a paper has already been processed using get_all_document_ids."""
        try:
            paper_id = paper.entry_id.split('/')[-1]
            existing_ids = self.db_interface.get_all_document_ids()
            is_duplicate = any(paper_id in doc_id for doc_id in existing_ids)
            
            if is_duplicate:
                self.logger.info(f"📝 Paper already processed: {paper.title}")
            else:
                self.logger.debug(f"Paper not found in database: {paper.title}")
                
            return is_duplicate
            
        except Exception as e:
            self.logger.error(f"Error checking for duplicates: {str(e)}")
            return False  # On error, assume not duplicate to allow processing
        
    def _create_search_query(self, search_terms: List[str]) -> str:
        """
        Create a more flexible search query that combines exact and partial matches.
        """
        # Create two types of terms:
        # 1. Exact phrases for highly specific matches
        # 2. AND combinations for broader matches
        exact_phrases = []
        broad_terms = []
        
        for term in search_terms:
            # Convert multi-word terms to AND queries
            words = term.split()
            if len(words) > 1:
                # Add both exact phrase and AND combination
                exact_phrases.append(f'"{term}"')
                broad_terms.append(" AND ".join(words))
            else:
                broad_terms.append(term)
        
        # Combine all terms with OR
        all_terms = exact_phrases + broad_terms
        combined_query = " OR ".join(all_terms)
        
        return combined_query

    def process_arxiv_papers(self, search_query: str):
        """Process arXiv papers with improved search strategy."""
        try:
            self.logger.info(f"🔍 Starting paper search and processing")
            
            # Split search terms and create optimized query
            search_terms = [term.strip() for term in search_query.split(" OR ")]
            optimized_query = self._create_search_query(search_terms)
            
            self.logger.info(f"📊 Using optimized search query: {optimized_query}")
            
            papers = self.arxiv_scraper.search_papers(
                optimized_query,
                max_results=self.config.processing.max_papers
            )
            
            if not papers:
                # Try a broader search if no results found
                broad_terms = [term.split()[-1] for term in search_terms]  # Use last word of each term
                broad_query = " OR ".join(broad_terms)
                self.logger.info(f"📊 Trying broader search: {broad_query}")
                
                papers = self.arxiv_scraper.search_papers(
                    broad_query,
                    max_results=self.config.processing.max_papers
                )
            
            if not papers:
                self.logger.warning("❌ No papers found matching the query")
                return
                
            # Filter out duplicates
            papers = [p for p in papers if not self._check_duplicate(p)]
            
            if not papers:
                self.logger.info("All found papers have already been processed")
                return
                
            self.stats.total_papers = len(papers)
            self.logger.info(f"📚 Found {len(papers)} new papers to process")
            
            for i in range(0, len(papers), self.config.processing.paper_batch_size):
                batch = papers[i:i + self.config.processing.paper_batch_size]
                self._process_paper_batch(batch)
                
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.stats.end_time = datetime.now()
            self._log_final_stats()

    def _process_paper_batch(self, papers: List):
        """Process a batch of papers with improved tracking."""
        with ThreadPoolExecutor(max_workers=self.config.processing.max_workers) as executor:
            futures = {
                executor.submit(self._download_and_process_paper, paper): paper
                for paper in papers
            }
            
            with tqdm(total=len(papers), desc="📄 Processing papers", ncols=80) as pbar:
                for future in as_completed(futures):
                    paper = futures[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            self.logger.info(f"Processing {len(chunks)} chunks from paper: {paper.title}")
                            if self._batch_add_to_db(chunks):
                                self.stats.processed_papers += 1
                                self.logger.info(f"✅ Successfully processed paper: {paper.title}")
                            else:
                                self.stats.failed_papers += 1
                                self.logger.error(f"❌ Failed to add chunks for paper: {paper.title}")
                        else:
                            self.stats.failed_papers += 1
                            self.logger.warning(f"⚠️ No chunks generated for paper: {paper.title}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to process paper {paper.title}: {str(e)}")
                        self.stats.failed_papers += 1
                    finally:
                        pbar.update(1)
                        
    def _download_and_process_paper(self, paper) -> Optional[List[DocumentChunk]]:
        """Download and process a single paper with improved error handling."""
        try:
            self.logger.info(f"📥 Downloading: '{paper.title}'")
            pdf_filepath = self.arxiv_scraper.download_paper_pdf(paper)
            
            if not pdf_filepath:
                self.logger.warning(f"❌ Failed to download: {paper.title}")
                return None
                
            # Get metadata from arxiv paper
            metadata_dict = self.arxiv_scraper.get_paper_metadata(paper)
            
            # Add source_document_id to metadata dict
            metadata_dict['source_document_id'] = paper.entry_id.split('/')[-1]
            
            # Create metadata object with all fields
            document_metadata = DocumentMetadata(**metadata_dict)
            
            chunks = self._process_pdf_document(pdf_filepath, document_metadata)
            if chunks:
                self.logger.info(f"✅ Successfully processed: {paper.title}")
                return chunks
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {paper.title}: {str(e)}")
            return None
    
    def _process_pdf_document(
            self,
            pdf_filepath: str,
            document_metadata: DocumentMetadata
        ) -> List[DocumentChunk]:
        """Process PDF document with improved chunk ID generation."""
        chunks = []
        try:
            base_id = document_metadata.source_document_id or document_metadata.source_document_title
            
            for i, chunk_text in enumerate(self._extract_text_generator(pdf_filepath)):
                chunk_metadata = document_metadata.model_copy(deep=True)
                chunk_metadata.chunk_id = f"{base_id}-chunk-{i+1}"
                
                chunks.append(
                    DocumentChunk(content=chunk_text, metadata=chunk_metadata)
                )
                
            self.logger.info(f"📄 Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ PDF processing error: {str(e)}")
            return []

    def _extract_text_generator(self, pdf_filepath: str) -> Generator[str, None, None]:
        with open(pdf_filepath, 'rb') as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            current_chunk = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                    
                current_chunk += page_text + "\n\n"
                
                while len(current_chunk) >= self.config.vector_db.chunk_size:
                    chunk_to_yield = current_chunk[:self.config.vector_db.chunk_size]
                    current_chunk = current_chunk[
                        self.config.vector_db.chunk_size -
                        self.config.vector_db.chunk_overlap:
                    ]
                    yield chunk_to_yield
                    
            if current_chunk:
                yield current_chunk

    def _batch_add_to_db(self, chunks: List[DocumentChunk]) -> bool:
            """
            Add chunks to database with improved error handling and batch management.
            Returns True if successful, False otherwise.
            """
            if not chunks:
                self.logger.warning("No chunks to add to database")
                return False

            try:
                batch_size = min(self.config.vector_db.batch_size, 10)  # Limit batch size
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                self.logger.info(f"Starting database insertion of {len(chunks)} chunks in {total_batches} batches")
                
                with tqdm(
                    total=total_batches, 
                    desc="💾 Adding to database", 
                    leave=False,
                    ncols=80
                ) as pbar:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        try:
                            # Add explicit error handling for database operation
                            self.logger.debug(f"Adding batch {i//batch_size + 1}/{total_batches}")
                            self.db_interface.add_documents(batch)
                            self.stats.successful_chunks += len(batch)
                            self.stats.total_chunks += len(batch)
                            pbar.update(1)
                        except Exception as e:
                            self.logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
                            self.stats.failed_chunks += len(batch)
                            continue
                
                if self.stats.successful_chunks > 0:
                    self.logger.info(f"✅ Successfully added {self.stats.successful_chunks} chunks to database")
                    return True
                else:
                    self.logger.error("❌ No chunks were successfully added to database")
                    return False

            except Exception as e:
                self.logger.error(f"❌ Database operation failed completely: {str(e)}\n{traceback.format_exc()}")
                return False

    def _process_paper_batch(self, papers: List):
            """
            Process a batch of papers with improved success tracking.
            """
            with ThreadPoolExecutor(max_workers=self.config.processing.max_workers) as executor:
                futures = {
                    executor.submit(self._download_and_process_paper, paper): paper 
                    for paper in papers
                }
                
                with tqdm(total=len(papers), desc="📄 Processing papers", ncols=80) as pbar:
                    for future in as_completed(futures):
                        paper = futures[future]
                        try:
                            chunks = future.result()
                            if chunks:
                                self.logger.info(f"Processing {len(chunks)} chunks from paper: {paper.title}")
                                if self._batch_add_to_db(chunks):
                                    self.stats.processed_papers += 1
                                    self.logger.info(f"✅ Successfully processed paper: {paper.title}")
                                else:
                                    self.stats.failed_papers += 1
                                    self.logger.error(f"❌ Failed to add chunks for paper: {paper.title}")
                            else:
                                self.stats.failed_papers += 1
                                self.logger.warning(f"⚠️ No chunks generated for paper: {paper.title}")
                        except Exception as e:
                            self.logger.error(f"❌ Failed to process paper {paper.title}: {str(e)}")
                            self.stats.failed_papers += 1
                        finally:
                            pbar.update(1)

    def _log_final_stats(self):
            """
            Enhanced final statistics logging.
            """
            duration = self.stats.end_time - self.stats.start_time
            success_rate = (self.stats.successful_chunks / self.stats.total_chunks * 100 
                        if self.stats.total_chunks > 0 else 0)
            
            self.logger.info(f"""
    📊 Processing Summary
    -------------------
    ⏱️  Duration: {duration}
    📚 Papers Processed: {self.stats.processed_papers}/{self.stats.total_papers}
    ❌ Failed Papers: {self.stats.failed_papers}
    📄 Chunks Successfully Added: {self.stats.successful_chunks}
    ❌ Failed Chunks: {self.stats.failed_chunks}
    💾 Total Chunks Processed: {self.stats.total_chunks}
    📈 Success Rate: {success_rate:.1f}%
    🔧 Peak Memory Usage: {self.stats.peak_memory_usage:.1f}%
            """)

if __name__ == "__main__":
    config = load_config()
    db_interface = VectorDBInterface(config)
    db_interface.initialize_db()
    
    processor = MemoryEfficientDocumentProcessor(config, db_interface)
    search_term = "Retrieval Augmented Generation"
    processor.process_arxiv_papers(search_term)

    # Command line argument to control database reset. Note: This will delete all data in the database.
    '''
    Run normally without resetting database:
        - python document_processor.py
    Run with database reset:
        - python document_processor.py --reset-db
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-db', action='store_true', help='Reset the database before processing')
    args = parser.parse_args()

    if args.reset_db:
        db_interface.reset_database()
