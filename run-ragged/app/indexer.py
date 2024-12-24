"""
RAG (Retrieval-Augmented Generation) Indexer Service

This service is responsible for processing PDF documents stored in Google Cloud Storage,
converting them into vector embeddings, and storing them in a PostgreSQL database using pgvector.
The service exposes APIs for health monitoring and on-demand indexing operations.

Key components:
- FastAPI web service for API exposure
- Google Cloud Storage integration for PDF retrieval
- PostgreSQL with pgvector for embedding storage
- Vertex AI for embedding generation
- SQLAlchemy for database operations

Environment Variables Required:
- DB_INSTANCE_NAME: Cloud SQL instance connection name
- DB_USER: Database username
- DB_PASS: Database password
- DB_NAME: Database name
- PDF_BUCKET_NAME: GCS bucket containing PDFs to be processed
"""

import os
import asyncio
import tempfile
from urllib.parse import quote_plus
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
from google.cloud.sql.connector import Connector
import pg8000
from sqlalchemy import Column, Integer, String, JSON, create_engine, MetaData, text
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List
import json
from datetime import datetime

# Initialize cloud services clients
storage_client = storage.Client()
connector = Connector()

# Create SQLAlchemy base class
Base = declarative_base(metadata=MetaData())

# Define Document Embedding model
class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    embedding = Column(Vector(768))  # Gecko model dimension
    doc_metadata = Column(JSON)
    
    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id})>"

def getconn() -> pg8000.dbapi.Connection:
    """Creates a connection to Cloud SQL PostgreSQL instance."""
    conn: pg8000.dbapi.Connection = connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )
    return conn

# Create database engine
engine = create_engine(
    "postgresql+pg8000://",
    creator=getconn
)

# Create embeddings instance
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

def init_vector_store(engine):
    """Initialize vector store tables and extensions"""
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(engine)

async def add_embeddings(engine, texts: List[str], metadatas: List[dict]):
    """Store text embeddings in the database"""
    # Get embeddings for all texts
    embedding_vectors = embeddings.embed_documents(texts)
    
    with Session(engine) as session:
        for text, metadata, vector in zip(texts, metadatas, embedding_vectors):
            embedding = DocumentEmbedding(
                content=text,
                embedding=vector,
                doc_metadata=metadata  # Changed from metadata to doc_metadata
            )
            session.add(embedding)
        session.commit()

def similarity_search(engine, query_text: str, k: int = 4):
    """Search for similar documents"""
    query_embedding = embeddings.embed_query(query_text)
    
    with Session(engine) as session:
        results = session.execute(
            text("""
                SELECT content, doc_metadata, 
                       1 - (embedding <=> :query_vector) as similarity
                FROM document_embeddings
                ORDER BY embedding <=> :query_vector
                LIMIT :k
            """),
            {
                "query_vector": query_embedding,
                "k": k
            }
        ).fetchall()
        
        return [
            {
                "content": r.content,
                "metadata": r.doc_metadata,  # Changed from metadata to doc_metadata in result
                "similarity": r.similarity
            }
            for r in results
        ]

async def process_pdf(blob_name: str) -> int:
    """Process a single PDF file from GCS bucket into vector embeddings."""
    bucket_name = os.getenv("PDF_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("PDF_BUCKET_NAME environment variable is not set")

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        blob.download_to_file(temp_file)
        temp_file_path = temp_file.name

    try:
        # Process PDF into text chunks
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

        # Prepare data for vector store
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": blob_name,
                "page": chunk.metadata["page"],
                "chunk_index": i
            } for i, chunk in enumerate(chunks)
        ]

        # Store embeddings
        await add_embeddings(engine, texts, metadatas)
        print(f"Processed {blob_name}: {len(texts)} chunks added")
        return len(texts)
    finally:
        os.unlink(temp_file_path)

async def index_pdfs(specific_files: list[str] = None):
    """Index multiple PDF files from the GCS bucket."""
    bucket_name = os.getenv("PDF_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("PDF_BUCKET_NAME environment variable is not set")

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    if specific_files:
        pdf_files = [name for name in specific_files if name.lower().endswith('.pdf')]
    else:
        pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]

    if not pdf_files:
        return {"message": "No PDF files found to index", "files_processed": 0}

    tasks = [process_pdf(pdf_file) for pdf_file in pdf_files]
    results = await asyncio.gather(*tasks)
    total_chunks = sum(results)

    return {
        "message": "Indexing complete",
        "files_processed": len(pdf_files),
        "total_chunks": total_chunks
    }

def init_feedback_table():
    """Initialize the feedback table in the database."""
    conn = getconn()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_feedback (
                id SERIAL PRIMARY KEY,
                message_id BIGINT NOT NULL,
                feedback_value TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                source_context JSONB,
                metadata JSONB
            );
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id
            ON message_feedback(message_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_feedback_value
            ON message_feedback(feedback_value);
        """)

        cursor.execute("""
            CREATE OR REPLACE VIEW feedback_analytics AS
            SELECT
                source_context->>'source' as document_source,
                feedback_value,
                COUNT(*) as feedback_count,
                AVG(CASE WHEN feedback_value = 'positive' THEN 1 ELSE 0 END) as positive_ratio
            FROM message_feedback
            WHERE source_context->>'source' IS NOT NULL
            GROUP BY document_source, feedback_value;
        """)

        conn.commit()
    except Exception as e:
        print(f"Failed to initialize feedback table: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

class FeedbackRequest(BaseModel):
    """Schema for feedback request"""
    message_id: int
    feedback_value: str
    source_context: Optional[dict] = None
    metadata: Optional[dict] = None

class IndexRequest(BaseModel):
    """Schema for index request payload"""
    files: list[str] | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        # Initialize vector store
        init_vector_store(engine)
        print("Vector store initialized successfully")
        
        # Initialize feedback table
        init_feedback_table()
        print("Feedback table initialized successfully")

        # Perform initial indexing
        await index_pdfs()
        yield
    except Exception as e:
        print(f"Startup initialization failed: {e}")
        yield
    finally:
        # Cleanup operations can go here
        pass

# Initialize FastAPI with CORS support and lifespan
app = FastAPI(
    title="RAG Indexer Service",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """API endpoint to record message feedback."""
    try:
        conn = getconn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO message_feedback
            (message_id, feedback_value, source_context, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (
                request.message_id,
                request.feedback_value,
                json.dumps(request.source_context) if request.source_context else None,
                json.dumps(request.metadata) if request.metadata else None
            )
        )

        conn.commit()
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/feedback/analytics")
async def get_feedback_analytics():
    """Get analytics about feedback patterns across different sources."""
    try:
        conn = getconn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM feedback_analytics;
        """)

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {
            "analytics": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/index")
async def trigger_indexing(request: IndexRequest | None = None):
    """API endpoint to trigger document indexing."""
    try:
        files = request.files if request and request.files else None
        result = await index_pdfs(files)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
