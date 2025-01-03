"""
Indexer Service for Processing PDF Documents and Managing Embeddings.

This service is responsible for:
    - Processing PDF documents stored in Google Cloud Storage (GCS).
    - Converting the text content of PDFs into vector embeddings using Vertex AI.
    - Storing the document embeddings in a PostgreSQL database with the pgvector extension.
    - Providing health check endpoints to monitor service status and connections.
    - Allowing feedback to be recorded and analyzed.

Key Components:
    - FastAPI: Web framework for building APIs.
    - Google Cloud Storage (GCS): Stores PDF documents.
    - Cloud SQL (PostgreSQL) with pgvector: Stores document embeddings.
    - Vertex AI: Provides embeddings and LLM capabilities.
    - Asynchronous Operations: Utilizes asyncio for non-blocking I/O operations.
    - Configuration: Uses environment variables for customizable settings.

Environment Variables (Required):
    - DB_INSTANCE_NAME: Cloud SQL instance connection name.
    - DB_NAME: Database name.
    - DB_PASS: Database password.
    - DB_USER: Database username.
    - PDF_BUCKET_NAME: GCS bucket name for PDF storage.
    - PRE_SHARED_KEY: Pre-shared key for authenticating with the server service.


Environment Variables (Optional):
    - CHUNK_OVERLAP: Overlap between text chunks for embedding (default: 50).
    - CHUNK_SIZE: Size of each text chunk for embedding (default: 2000).
    - EMBEDDING_BATCH_SIZE: Number of embeddings to process in a single batch (default: 10).
    - MAX_CHUNKS: Maximum number of chunks to process in a single batch (default: 50).
    - MAX_FILE_SIZE: Maximum file size in bytes for PDF processing (default: 100MB).
"""

import asyncio
import json
import logging
import logging.config
import os
import tempfile
from app.logging_config import setup_logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import BackgroundTasks, Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader
from typing import Any, Dict, Final, List, Optional

# Environment-based configuration (with more robust defaults and error handling)
CHUNK_OVERLAP: Final = int(os.getenv('CHUNK_OVERLAP', '50'))
CHUNK_SIZE: Final = int(os.getenv('CHUNK_SIZE', '2000'))
EMBEDDING_BATCH_SIZE: Final = int(os.getenv('EMBEDDING_BATCH_SIZE', '10'))
MAX_CHUNKS: Final = int(os.getenv('MAX_CHUNKS', '50'))
MAX_FILE_SIZE: Final = int(os.getenv('MAX_FILE_SIZE', str(100 * 1024 * 1024)))  # 100MB default, convert string to int
MAX_FILE_SIZE = int(MAX_FILE_SIZE) # convert str to int
PRE_SHARED_KEY: Final = str(os.getenv("PRE_SHARED_KEY"))

# Initialize logging with appropriate level
setup_logging()
logger = logging.getLogger(__name__)

# Initialize cloud services clients
storage_client = storage.Client()
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

# Initialize database connector
connector = Connector()

class DatabaseManager:
    """Manages database operations and connections."""
    def __init__(self):
        """Initializes the DatabaseManager with connection locks and status."""
        logger.info("Initializing DatabaseManager")
        self._startup_lock = asyncio.Lock()
        self._indexing_lock = asyncio.Lock()
        self.is_indexing = False
        self.conn = None
        # self.pool_size = 10 # Remove pool_size
      
    @asynccontextmanager
    async def get_connection(self):
        """
        Provides an async context manager for database connections.
        Ensures connection is healthy and properly closed.
        """
        if not self.conn:
            logger.info("Establishing a new database connection.")
            await self.ensure_connection()  # Call ensure_connection directly
        try:
            await self.ensure_connection()
            yield self.conn
        finally:
            # No explicit close here. AsyncPG's pool will handle it
            pass # we dont close here because the connection comes from pool
          
    async def ensure_connection(self):
        """
        Ensures the database connection is alive. Reconnects if necessary.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        try:
            if not self.conn:
                logger.info("No database connection exists, attempting to establish a new connection")
                try:
                    params = self._get_connection_params()
                    # Use connect_async with correct parameter names
                    conn = await connector.connect_async(
                        instance_connection_string=params["DB_INSTANCE_NAME"],
                        driver="asyncpg",
                        user=params["DB_USER"],
                        password=params["DB_PASS"],
                        db=params["DB_NAME"],
                        ip_type=IPTypes.PUBLIC,
                    )
                    self.conn = conn
                    logger.info("Database connection established successfully")
                except Exception as e:
                    logger.error(f"Failed to establish database connection: {e}")
                    return False
            try:
                # Test the connection with a simple+lightweight query
                await self.conn.execute('SELECT 1')
                return True
            except Exception as e:
                logger.warning(f"Database connection test failed: {e}, attempting reconnection")
                try:
                    await self.close()  # Close the old connection
                except:
                    pass  # Ignore errors on close
                return False

        except Exception as e:
            logger.error(f"Failed to ensure database connection: {str(e)}")
            return False

    def _get_connection_params(self) -> Dict[str, str]:
        """
         Retrieves database connection parameters from environment variables.

         Returns:
             Dict[str, str]: Database connection parameters.

         Raises:
             ValueError: If a required environment variable is missing.
         """
        required_params = ["DB_USER", "DB_PASS", "DB_NAME", "DB_INSTANCE_NAME"]
        params = {}

        for param in required_params:
            value = os.getenv(param)
            if not value:
                raise ValueError(f"Missing required environment variable: {param}")
            params[param] = value

        return params

    async def add_embeddings(self, texts: List[str], metadatas: List[Dict]) -> int:
        """
        Adds embeddings to the database.

        Args:
            texts (List[str]): List of text chunks to embed.
            metadatas (List[Dict]): List of metadata dictionaries.

        Returns:
            int: The number of successfully embedded chunks.
        """
        try:
          async with self.get_connection() as conn:
            # Prepare data for batch insert with explicit type casting
            values = [
                (
                    text,
                    str(await asyncio.to_thread(embeddings.embed_query, text)),  # Embed on a background thread, convert to string
                    json.dumps(metadata)
                )
                for text, metadata in zip(texts, metadatas) if text.strip()
             ]

            # Use executemany for batch insertion, and jsonb casting
            await conn.executemany(
                """
                INSERT INTO document_embeddings (content, embedding, doc_metadata)
                VALUES ($1, $2, $3::jsonb)
                """,
                values
            )

            logger.info(f"Inserted {len(values)} embeddings successfully")
            return len(values)
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}", exc_info=True)
            return 0

    async def close(self):
        """Closes the database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed.")
            self.conn = None

    async def initialize(self):
         """
         Initializes the database schema, including creating the pgvector extension
         and the necessary tables for document embeddings and message feedback.
         """
         async with self._startup_lock:
            try:
               async with self.get_connection() as conn:
                # Enable pgvector and create tables if not exists
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS document_embeddings (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        embedding vector(768),
                        doc_metadata JSONB
                    );
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS message_feedback (
                        id SERIAL PRIMARY KEY,
                        message_id BIGINT NOT NULL,
                        feedback_value TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        source_context JSONB,
                        metadata JSONB
                    );
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id
                    ON message_feedback(message_id);
                """)
                logger.info("Database initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
    
    def check_document_is_indexed(self, source: str) -> bool:
        """
        Checks if a document (by source) is already fully indexed by looking for
        the `is_embedded` metadata flag.

        Args:
            source (str): The source of the document

        Returns:
            bool: True if a document is indexed, False otherwise.
        """
        conn = None
        try:
            params = self._get_connection_params()
            # Create connection dictionary
            conn_kwargs = {
                "user": params["DB_USER"],
                "password": params["DB_PASS"],
                "db": params["DB_NAME"],
            }

            # Get connection from Cloud SQL Connector
            conn = connector.connect(
               params["DB_INSTANCE_NAME"],
               "pg8000",
               **conn_kwargs,
               ip_type=IPTypes.PUBLIC
            )
            
            cursor = conn.cursor()
            cursor.execute(
                """
                 SELECT EXISTS (
                    SELECT 1
                    FROM document_embeddings
                    WHERE doc_metadata->>'source' = %s
                    AND doc_metadata->>'is_embedded' = 'true'
                 )
                """,
                 (source,)
            )
            result = cursor.fetchone()[0]
            cursor.close()
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking if document is indexed: {str(e)}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()


# Initialize database manager
db = DatabaseManager()

async def process_pdf(blob_name: str) -> int:
     """
     Processes a single PDF file from GCS bucket into vector embeddings.
     Args:
        blob_name (str): The name of the blob to process
     Returns:
        int: The number of embedded document chunks
     """
     bucket_name = os.getenv("PDF_BUCKET_NAME")
     if not bucket_name:
         logger.error("PDF_BUCKET_NAME environment variable is not set")
         raise ValueError("PDF_BUCKET_NAME environment variable is not set")

     bucket = storage_client.bucket(bucket_name)
     blob = bucket.blob(blob_name)

     try:
         # Check file size and log it
         blob.reload()
         logger.info(f"Processing PDF: {blob_name} (Size: {blob.size/1024/1024:.2f}MB)")

         if hasattr(blob, 'size') and blob.size > MAX_FILE_SIZE:
             logger.warning(f"File {blob_name} exceeds size limit ({blob.size/1024/1024:.2f}MB > {MAX_FILE_SIZE/1024/1024}MB)")
             return 0

         # Check if document is already indexed (with the new metadata field)
         if db.check_document_is_indexed(blob_name):
             logger.info(f"Document {blob_name} already fully indexed, skipping.")
             return 0

         # Use a context manager for temporary file handling
         with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_file:
             logger.info(f"Starting download of {blob_name}")
             try:
                 blob.download_to_file(temp_file)
                 temp_file.flush()  # Ensure all data is written
                 logger.info(f"Successfully downloaded {blob_name}")
             except Exception as e:
                 logger.error(f"Failed to download {blob_name}: {str(e)}", exc_info=True)
                 return 0
             try:
                 logger.info(f"Processing PDF content from {blob_name}")

                 async def process_pdf_chunk(start_page: int, end_page: int, pdf_reader) -> List[Document]:
                     """Process a chunk of PDF pages."""
                     pages = []
                     for page_num in range(start_page, min(end_page, len(pdf_reader.pages))):
                         try:
                             page = pdf_reader.pages[page_num]
                             text = page.extract_text()
                             if text.strip():
                                  pages.append(Document(
                                      page_content=text,
                                      metadata={
                                          "source": blob_name.replace(".pdf", ""),
                                          "page": page_num
                                      }
                                  ))
                             logger.info(f"Extracted text from page {page_num + 1}")
                         except Exception as e:
                             logger.error(f"Error extracting text from page {page_num}: {str(e)}")
                             continue
                     return pages

                 # Process PDF in chunks to manage memory
                 pdf_reader = PdfReader(temp_file.name)
                 num_pages = len(pdf_reader.pages)
                 logger.info(f"PDF has {num_pages} pages")

                 pages = []
                 chunk_size = 10  # Process 10 pages at a time

                 # Removed asyncio.timeout around loop
                 for start in range(0, num_pages, chunk_size):
                     end = min(start + chunk_size, num_pages)
                     logger.info(f"Processing pages {start + 1} to {end} of {num_pages}")

                     chunk_pages = await process_pdf_chunk(start, end, pdf_reader)
                     pages.extend(chunk_pages)

                     logger.info(f"Processed {len(chunk_pages)} pages in current chunk. "
                               f"Total pages processed: {len(pages)}")

                 if not pages:
                     logger.error(f"No text content extracted from {blob_name}")
                     return 0

                 # Create text chunks with optimized parameters
                 logger.info(f"Starting text splitting for {blob_name}")
                 text_splitter = RecursiveCharacterTextSplitter(
                     chunk_size=CHUNK_SIZE,
                     chunk_overlap=CHUNK_OVERLAP,
                     length_function=len,
                 )
                 chunks = await asyncio.to_thread(text_splitter.split_documents, pages)
                 total_chunks = len(chunks)
                 logger.info(f"Split {blob_name} into {total_chunks} chunks")

                 # Process chunks in groups
                 total_processed = 0
                 for chunk_start in range(0, total_chunks, MAX_CHUNKS):
                     chunk_end = min(chunk_start + MAX_CHUNKS, total_chunks)
                     current_chunks = chunks[chunk_start:chunk_end]

                     texts = []
                     metadatas = []

                     for i, chunk in enumerate(current_chunks):
                         if chunk.page_content.strip():
                             texts.append(chunk.page_content)
                             metadatas.append({
                                 "source": chunk.metadata.get("source", None),  # Keep the source
                                 "page": chunk.metadata.get("page", 0),
                                 "chunk_index": i + chunk_start,
                                 "chunk_length": len(chunk.page_content),
                                 "processed_at": datetime.utcnow().isoformat(),
                                 "chunk_group": f"{chunk_start}-{chunk_end}"
                             })


                     logger.info(f"Processing chunk batch {chunk_start} to {chunk_end}")
                     if texts:
                         try:
                             #Move is_embedded update inside the transaction
                             chunks_added = await db.add_embeddings(texts, metadatas)

                             if chunks_added > 0:
                                try:
                                  async with db.get_connection() as conn:
                                   await conn.execute(
                                        """
                                        UPDATE document_embeddings
                                        SET doc_metadata = jsonb_set(doc_metadata, '{is_embedded}', 'true'::jsonb)
                                        WHERE doc_metadata->>'source' = $1
                                        """,
                                        blob_name.replace(".pdf", "")
                                     )
                                  logger.info(f"Successfully marked {blob_name} as fully embedded.")
                                except Exception as e:
                                   logger.error(f"Error setting document 'is_embedded' flag after processing {blob_name}: {str(e)}", exc_info=True)

                             total_processed += chunks_added
                             logger.info(f"Successfully processed {chunks_added} chunks from batch {chunk_start}-{chunk_end}")
                         except Exception as e:
                             logger.error(f"Error processing chunks {chunk_start}-{chunk_end}: {str(e)}")
                             continue

                     # Add a delay between chunk groups if not the last group
                     if chunk_end < total_chunks:
                         logger.info("Waiting before processing next chunk group...")
                         await asyncio.sleep(5)

                 logger.info(f"Completed processing {total_processed} chunks for {blob_name}")
                 return total_processed

             except asyncio.TimeoutError:
                 logger.error(f"Timeout while processing {blob_name}")
                 return 0
             except Exception as e:
                 logger.error(f"Error processing {blob_name}: {str(e)}", exc_info=True)
                 return 0

     except Exception as e:
         logger.error(f"Failed to process {blob_name}: {str(e)}", exc_info=True)
         return 0

# Request models
class FeedbackRequest(BaseModel):
    """Schema for feedback request."""
    message_id: int
    feedback_value: str
    source_context: Optional[dict] = None
    metadata: Optional[dict] = None

class IndexRequest(BaseModel):
    """Schema for index request payload."""
    files: Optional[List[str]] = Field(default=None, description="List of files to index")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await db.initialize()
        yield
    finally:
        if db.conn:
           await db.close()

# Initialize FastAPI with CORS support and lifespan
app = FastAPI(
    title="RAG Indexer Service",
    lifespan=lifespan,
    docs_url="/api" # This is important since the default is /docs which conflicts with sphinx
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def psk_middleware(request: Request, call_next):
    """Middleware to check for a valid pre-shared key."""
    if request.url.path == "/health":
        return await call_next(request) # allow health checks without PSK

    provided_key = request.headers.get("X-Pre-Shared-Key")
    if not provided_key or provided_key != PRE_SHARED_KEY:
        logger.warning(f"Request to {request.url.path} rejected due to invalid pre-shared key. Provided key: {provided_key}")
        raise HTTPException(
             status_code=403, detail="Invalid or missing pre-shared key"
         )

    return await call_next(request)

app.middleware("http")(psk_middleware)


# API endpoints
@app.get('/favicon.ico')
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    try:
        return {
            "status": "healthy",
            "indexing_status": "running" if db.is_indexing else "idle",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/status")
async def get_indexing_status():
    """Get current indexing status."""
    return {
        "is_indexing": db.is_indexing,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/debug/reset-embeddings")
async def reset_embeddings(request: Request):
 """WARNING: Development only - Resets all embeddings data"""
 # Get collaboration settings from the request
 try:
     # body = await request.json()
     # debug_mode = body.get("debug_mode", False)
     # if not debug_mode:
     #    raise HTTPException(
     #        status_code=403,
     #        detail="This endpoint is only available when debug mode is enabled in collaboration settings"
     #    )

    logger.warning("Debug Mode: Executing complete embeddings reset")
    async with db.get_connection() as conn:
        await conn.execute("DELETE FROM document_embeddings")
        await conn.execute("ALTER SEQUENCE document_embeddings_id_seq RESTART WITH 1")
    logger.info("Database reset completed successfully")
    return {
         "message": "Database reset completed",
         "timestamp": datetime.utcnow().isoformat(),
         "warning": "Debug mode operation completed"
     }
 except Exception as e:
     logger.error(f"Reset failed: {e}")
     raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """Record message feedback."""
    try:
        # Removed direct db call to use connection context
        async with db.get_connection() as conn:
            await conn.execute(
            """
                INSERT INTO message_feedback (message_id, feedback_value, source_context, metadata)
                VALUES ($1, $2, $3::jsonb, $4::jsonb)
            """,
             request.message_id,
             request.feedback_value,
             json.dumps(request.source_context),
             json.dumps(request.metadata)
         )
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint that checks all connections and counts.
    """
    try:
       async with db.get_connection() as conn:
           # Get detailed document counts
           detailed_counts = await conn.fetch("""
               SELECT
                   COUNT(*) as total_docs,
                   COUNT(CASE WHEN doc_metadata->>'source' IS NOT NULL THEN 1 END) as docs_with_source,
                   COUNT(CASE WHEN doc_metadata->>'content_hash' IS NOT NULL THEN 1 END) as docs_with_hash,
                   COUNT(CASE WHEN doc_metadata->>'is_embedded' = 'true' THEN 1 END) as docs_marked_embedded,
                   COUNT(DISTINCT doc_metadata->>'source') as unique_sources
               FROM document_embeddings
           """)

           counts = dict(detailed_counts[0])

           # Check feedback counts
           feedback_count = await conn.fetchval("SELECT COUNT(*) FROM message_feedback")

           # Verify GCS bucket access and get PDF count
           bucket_name = os.getenv("PDF_BUCKET_NAME")
           bucket = storage_client.bucket(bucket_name)
           pdf_count = len([b for b in bucket.list_blobs() if b.name.lower().endswith('.pdf')])

           # Fetch 5 documents from embeddings table for inspection
           sample_docs = await conn.fetch("SELECT id, content, doc_metadata FROM document_embeddings LIMIT 5")
           sample_docs_list = [dict(row) for row in sample_docs]

           # Fetch Document Summary and Source Stats
           doc_summary = await conn.fetch("""
               SELECT
                 COUNT(*) as total_embeddings,
                 COUNT(DISTINCT doc_metadata->>'source') as unique_sources,
                 COUNT(CASE WHEN doc_metadata->>'is_embedded' = 'true' THEN 1 END) as embedded_docs
               FROM document_embeddings;
           """)
           doc_summary = dict(doc_summary[0])

           return {
               "status": "healthy",
               "db_connected": True,
               "bucket_available": True,
               "document_summary": doc_summary,
               "document_counts": {
                   "total_embeddings": counts["total_docs"],
                   "with_source": counts["docs_with_source"],
                   "with_hash": counts["docs_with_hash"],
                   "marked_embedded": counts["docs_marked_embedded"],
                   "unique_sources": counts["unique_sources"]
               },
               "sample_documents": sample_docs_list,
               "storage": {
                   "pdf_count": pdf_count
               },
               "feedback_count": feedback_count,
               "indexing_status": "running" if db.is_indexing else "idle"
           }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Modified trigger_indexing function
@app.post("/index")
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Trigger document indexing."""
    if db.is_indexing:
        return {"status": "already_running", "message": "Indexing operation already in progress"}

    try:
        files = request.files if request.files else None
        if files:
            logger.info(f"Starting indexing for specific files: {files}")
        else:
            logger.info("Starting indexing for all PDF files")

        db.is_indexing = True  # Set indexing to true *before* starting tasks

        async def run_indexing_tasks():
            """Helper function to execute and track background tasks."""
            try:
                tasks = []  # List to hold the task *coroutines*

                if files:
                     for file_name in files:
                        tasks.append(process_pdf(file_name))
                else:
                     bucket_name = os.getenv("PDF_BUCKET_NAME")
                     bucket = storage_client.bucket(bucket_name)
                     for blob in bucket.list_blobs():
                         if blob.name.lower().endswith('.pdf'):
                            tasks.append(process_pdf(blob.name))

                # Use asyncio.gather to run and wait for all tasks
                if tasks:
                   await asyncio.gather(*tasks)  # wait until all processing tasks are done
            finally:
                db.is_indexing = False  # Reset indexing status *after* all tasks finish
                logger.info("Indexing completed, status reset to idle.")

        background_tasks.add_task(run_indexing_tasks) # run in background
        return {"status": "started", "message": "Indexing started in background"}

    except Exception as e:
        logger.error(f"Failed to trigger indexing: {e}")
        db.is_indexing = False  # Reset indexing to false on failure
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)