# vector_db_interface.py (Updated)
import chromadb
from chromadb.utils import embedding_functions
from config import Config
from models import DocumentChunk, DocumentMetadata
from typing import List, Optional, Tuple
from utils import logger, time_it
import time
import traceback
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil # Import psutil if not already there - for memory tracking

class VectorDBInterface:
    def __init__(self, config: Config):
        self.config = config
        self.db_type = config.vector_db.db_type
        self.persist_directory = config.vector_db.persist_directory
        self.client = None
        self.collection = None
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        self.max_retries = config.vector_db.max_retries # Use config value
        self.retry_delay = config.vector_db.retry_delay # Use config value
        self.batch_size = config.vector_db.batch_size # Use config value from config

    def _retry_operation(self, operation, *args, operation_name="", **kwargs):
        """Enhanced retry mechanism with operation naming."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"⚠️ {operation_name} attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                continue
        logger.error(f"❌ {operation_name} failed after {self.max_retries} attempts: {str(last_exception)}")
        raise last_exception

    @time_it
    def initialize_db(self):
        """Initializes the vector database client and collection."""
        if self.db_type.lower() == "chroma":
            self._initialize_chroma()
        elif self.db_type.lower() == "qdrant":
            logger.warning("Qdrant support not yet implemented. Using Chroma.")
            self._initialize_chroma()
        else:
            logger.error(f"Unsupported database type: {self.db_type}. Using Chroma.")
            self._initialize_chroma()

    def _initialize_chroma(self):
        """Initializes ChromaDB with correct settings format."""
        try:
            logger.info(f"Initializing ChromaDB in: {self.persist_directory}")

            def init_client():
                # Configure ChromaDB settings using the correct format
                settings = chromadb.Settings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )

                # Create client with settings
                self.client = chromadb.PersistentClient(settings=settings)

                # Create or get collection
                self.collection = self.client.get_or_create_collection(
                    name="research_agent_collection",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )

            self._retry_operation(init_client, operation_name="Initialize ChromaDB")
            logger.info("✅ ChromaDB initialized successfully")

        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with error handling."""
        try:
            embeddings = self.embedding_function(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    @time_it
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add documents with cleaned metadata."""
        if not self.collection:
            raise ValueError("Database not initialized")

        if not chunks:
            logger.warning("No chunks to process")
            return

        try:
            logger.info(f"Processing {len(chunks)} chunks...")

            # Prepare all data first with cleaned metadata
            embeddings, ids, documents, metadatas = self._prepare_chunk_data(chunks)

            if not ids:
                logger.error("No valid chunks after preparation")
                return

            # Process in smaller batches
            successful_chunks = 0
            batch_size = self.batch_size # Use self.batch_size from config
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))

                try:
                    def add_batch():
                        logger.debug(f"Memory usage before adding batch: {psutil.virtual_memory().percent}%") # Example memory tracking log
                        self.collection.add(
                            embeddings=embeddings[i:batch_end],
                            ids=ids[i:batch_end],
                            documents=documents[i:batch_end],
                            metadatas=metadatas[i:batch_end]
                        )
                        logger.debug(f"Memory usage after adding batch: {psutil.virtual_memory().percent}%") # Example memory tracking log
                    self._retry_operation(
                        add_batch,
                        operation_name=f"Add batch {i//batch_size + 1}"
                    )
                    successful_chunks += batch_end - i
                    logger.info(f"✅ Added batch {i//batch_size + 1} ({batch_end - i} chunks)")

                except Exception as e:
                    logger.error(f"❌ Failed to add batch {i//batch_size + 1}: {str(e)}")
                    continue

            logger.info(f"✅ Successfully added {successful_chunks}/{len(chunks)} chunks")
            return successful_chunks > 0

        except Exception as e:
            logger.error(f"❌ Document addition failed: {str(e)}\n{traceback.format_exc()}")
            return False

    @time_it
    def search_similarity(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Perform similarity search with improved error handling."""
        if not self.collection:
            raise ValueError("Database not initialized")

        try:
            results = self._retry_operation(
                lambda: self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["metadatas", "documents"]
                ),
                operation_name="Similarity search"
            )

            chunks = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    try:
                        metadata_dict = results['metadatas'][0][i]

                        # Check if authors is a string and convert to a list if necessary
                        if isinstance(metadata_dict.get('authors'), str):
                            metadata_dict['authors'] = [metadata_dict['authors']]

                        metadata = DocumentMetadata(**metadata_dict)
                        chunk = DocumentChunk(
                            content=results['documents'][0][i],
                            metadata=metadata
                        )
                        chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Failed to process result {i}: {str(e)}")
                        continue

            return chunks

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_collection_size(self) -> int:
        """Get the current size of the collection."""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except Exception as e:
            logger.error(f"Failed to get collection size: {str(e)}")
            return 0

    # Add these methods to your VectorDBInterface class

    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the collection."""
        try:
            if not self.collection:
                logger.warning("Database not initialized")
                return []
                
            # Use peek() instead of get() to retrieve IDs
            results = self._retry_operation(
                lambda: self.collection.peek(limit=10000),  # Adjust limit based on your needs
                operation_name="Get all document IDs"
            )
            
            return results['ids'] if results and 'ids' in results else []
            
        except Exception as e:
            logger.error(f"Failed to get document IDs: {str(e)}")
            return []

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document with given ID exists in the collection."""
        try:
            if not self.collection:
                return False
                
            # Use get() with 'documents' in include parameter
            results = self._retry_operation(
                lambda: self.collection.get(
                    ids=[doc_id],
                    include=['documents']
                ),
                operation_name=f"Check document existence: {doc_id}"
            )
            
            return bool(results and 'documents' in results and results['documents'])
            
        except Exception as e:
            logger.error(f"Failed to check document existence: {str(e)}")
            return False

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """Retrieve a document by its ID."""
        try:
            if not self.collection:
                return None
                
            results = self._retry_operation(
                lambda: self.collection.get(
                    ids=[doc_id],
                    include=['documents', 'metadatas']
                ),
                operation_name=f"Get document: {doc_id}"
            )
            
            if (results and 'documents' in results and results['documents'] 
                and 'metadatas' in results and results['metadatas']):
                metadata_dict = results['metadatas'][0]
                
                # Handle authors conversion if needed
                if isinstance(metadata_dict.get('authors'), str):
                    metadata_dict['authors'] = [metadata_dict['authors']]
                    
                metadata = DocumentMetadata(**metadata_dict)
                return DocumentChunk(
                    content=results['documents'][0],
                    metadata=metadata
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            return None

    def _prepare_chunk_data(self, chunks: List[DocumentChunk]) -> Tuple[List, List, List, List]:
        """Prepare chunk data with duplicate checking."""
        embeddings = []
        ids = []
        documents = []
        metadatas = []
        
        # Get existing IDs for duplicate checking
        existing_ids = set(self.get_all_document_ids())
        logger.info(f"Found {len(existing_ids)} existing documents")
        
        # Process in smaller batches
        batch_size = min(self.batch_size, len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self._generate_embeddings_batch(batch_texts)
                
                # Process successful embeddings
                for j, chunk in enumerate(batch):
                    try:
                        clean_metadata = chunk.metadata.clean_for_chroma()
                        
                        # Generate a unique ID if not provided
                        chunk_id = (chunk.metadata.chunk_id if chunk.metadata.chunk_id
                                else f"{clean_metadata['source_document_title']}-chunk-{i+j+1}")
                        
                        # Skip if document already exists
                        if chunk_id in existing_ids:
                            logger.debug(f"Skipping existing document: {chunk_id}")
                            continue
                        
                        embeddings.append(batch_embeddings[j])
                        ids.append(chunk_id)
                        documents.append(chunk.content)
                        metadatas.append(clean_metadata)
                        
                    except Exception as e:
                        logger.error(f"Failed to process chunk {i+j}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size}: {str(e)}")
                continue
                
        if not embeddings:
            logger.error("No valid chunks prepared for database insertion")
        else:
            logger.info(f"Successfully prepared {len(embeddings)} new chunks for insertion")
            
        return embeddings, ids, documents, metadatas 

    def reset_database(self):
        """Reset database with improved error handling."""
        try:
            if self.client and self.collection:
                self._retry_operation(
                    lambda: self.client.delete_collection("research_agent_collection"),
                    operation_name="Reset database"
                )
                self.collection = None
                logger.info("✅ Database reset successful")
                self.initialize_db()
        except Exception as e:
            logger.error(f"❌ Database reset failed: {str(e)}")
            raise



if __name__ == "__main__":
    from config import load_config
    config = load_config()
    db_interface = VectorDBInterface(config)
    db_interface.initialize_db()

    # Test data
    chunks_to_add = [
        DocumentChunk(
            content="Large language models are great.",
            metadata=DocumentMetadata(
                source_document_title="Test Doc 1", 
                authors=["Author A"],
                chunk_id="test-doc-1-chunk-1"
            )
        ),
        DocumentChunk(
            content="Vector databases store embeddings.",
            metadata=DocumentMetadata(
                source_document_title="Test Doc 2", 
                authors=["Author B"],
                chunk_id="test-doc-2-chunk-1"
            )
        )
    ]

    # Test operations
    print("\nTesting database operations:")
    print("-" * 30)
    
    db_interface.reset_database()
    db_interface.initialize_db()
    
    # Test document addition
    print("\nTesting document addition:")
    success = db_interface.add_documents(chunks_to_add)
    print(f"Documents added successfully: {success}")
    print(f"Collection Size: {db_interface.get_collection_size()}")
    
    # Test document retrieval
    print("\nTesting document retrieval:")
    doc = db_interface.get_document_by_id("test-doc-1-chunk-1")
    if doc:
        print(f"Retrieved document: {doc.content}")
        print(f"Document metadata: {doc.metadata}")
    
    # Test search
    print("\nTesting similarity search:")
    results = db_interface.search_similarity("language models", top_k=2)
    for chunk in results:
        print(f"\nContent: {chunk.content}")
        print(f"Metadata: {chunk.metadata}")
    
    # Test document existence
    print("\nTesting document existence:")
    exists = db_interface.document_exists("test-doc-1-chunk-1")
    print(f"Document exists: {exists}")
    
    # Test cleanup
    print("\nTesting database reset:")
    db_interface.reset_database()
    print(f"Collection Size after reset: {db_interface.get_collection_size()}")