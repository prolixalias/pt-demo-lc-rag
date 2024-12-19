import os
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, HTTPException, Response, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.pgvector import PGVector
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from circuitbreaker import circuit, CircuitBreaker

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with CORS
app = FastAPI(title="Albear Einstein")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
PDF_PROCESSING_LATENCY = Histogram('pdf_processing_duration_seconds', 'PDF processing latency')
QUERY_LATENCY = Histogram('query_duration_seconds', 'Query processing latency')

# Circuit breaker setup
class ServiceCircuitBreaker:
    def __init__(self):
        self.db_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.storage_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.vertex_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

circuit_breakers = ServiceCircuitBreaker()

# Database connection
connector = Connector()

@circuit_breakers.db_breaker
def get_db_connection() -> pg8000.dbapi.Connection:
    """Create a database connection with circuit breaker protection."""
    try:
        conn = connector.connect(
            os.getenv("DB_INSTANCE_NAME", ""),
            "pg8000",
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASS", ""),
            db=os.getenv("DB_NAME", ""),
            ip_type=IPTypes.PUBLIC
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

# Initialize vector store
vectorstore = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args={"creator": get_db_connection},
    embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@003")
)

# Initialize LLM and chain
llm = VertexAI(
    model_name="gemini-1.0-pro-001",
    temperature=0.2,
    max_output_tokens=256,
    top_k=40,
    top_p=0.95
)

# Create retriever chain
def format_docs(docs):
    """Format retrieved documents with source information."""
    return "\n\n".join(f"[Source: {doc.metadata.get('source', 'Unknown')}, "
                      f"Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
                      for doc in docs)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create prompt template
prompt_template = PromptTemplate.from_template(
"""You are an AI assistant specialized in cybersecurity, IT governance, and legal compliance, tasked with answering questions based on the provided PDF documents in addition to your initial training. These documents include standards from CMMC, NIST, DoD, and various corporate IT policies. Your role is to:
    1) Analyze the retrieved document sections with a deep understanding of IT concepts, cybersecurity frameworks, and legal terminology.
    2) Interpret and synthesize information from these documents to provide accurate, context-aware responses to queries.
    3) Pay special attention to compliance requirements, security controls, and technical standards as described by these governing bodies and corporate policies.
    4) Use the following criteria to answer questions, ensuring you:
        a) Explain technical concepts in layman's terms where possible, but maintain precision for technical audiences.
        b) Cite specific documents, sections or clauses when answering questions.
        c) Highlight any discrepancies or overlapping requirements between different standards whenever relevant.
If you are unable to locate an answer within these documents, please state clearly that the information is not available in the set but answer using initial training. Remember, accuracy and adherence to your role are paramount.

Retrieved sections:
{context}

Question: {query}

Answer: """)

#prompt_template = PromptTemplate.from_template(
#"""You are an AI assistant.
#
#Retreived sections:
#{context}
#
#Question: {query}
#
#Answer: """)

# Chain everything together
chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "query": RunnablePassthrough()
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

# FastAPI routes
@app.get("/")
async def redirect_root_to_ui():
    """Redirect root to UI."""
    return RedirectResponse("/ui")

os.makedirs("frontend/dist/assets", exist_ok=True)
os.makedirs("frontend/public/assets", exist_ok=True)
app.mount("/ui/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
app.mount("/assets", StaticFiles(directory="frontend/public/assets"), name="assets")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def get_circuit_state(breaker: CircuitBreaker) -> str:
    """Get the current state of a circuit breaker."""
    return breaker.state.lower()

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "circuit_breakers": {
            "database": get_circuit_state(circuit_breakers.db_breaker),
            "storage": get_circuit_state(circuit_breakers.storage_breaker),
            "vertex": get_circuit_state(circuit_breakers.vertex_breaker)
        }
    }

@app.get("/files")
@circuit_breakers.storage_breaker
async def list_files():
    """List all PDF files in the GCS bucket."""
    try:
        bucket_name = os.getenv("PDF_BUCKET_NAME")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="PDF_BUCKET_NAME environment variable not set")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        files = []
        for blob in bucket.list_blobs():
            if blob.name.lower().endswith('.pdf'):
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "created": blob.time_created.isoformat(),
                    "updated": blob.updated.isoformat()
                })
                
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
@circuit_breakers.storage_breaker
async def upload_pdf(file: UploadFile):
    """Handle PDF file uploads to GCS."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    start_time = datetime.now()
    try:
        bucket_name = os.getenv("PDF_BUCKET_NAME")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="PDF_BUCKET_NAME environment variable not set")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file.filename)
        
        content = await file.read()
        blob.upload_from_string(content, content_type='application/pdf')
        
        return {
            "message": f"Successfully uploaded {file.filename}",
            "status": "success",
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PDF_PROCESSING_LATENCY.observe((datetime.now() - start_time).total_seconds())

@app.delete("/files/{filename}")
@circuit_breakers.storage_breaker
async def delete_file(filename: str):
    """Delete a PDF file and its associated vectors from storage"""
    try:
        # Delete from GCS
        bucket_name = os.getenv("PDF_BUCKET_NAME")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="PDF_BUCKET_NAME environment variable not set")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.delete()

        # Delete from vector store
        # Assuming you store the source in metadata
        vectorstore.delete({"source": filename})

        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
@circuit_breakers.vertex_breaker
async def query_documents(query: dict):  # Change parameter to accept dict
    """Query documents using RAG."""
    start_time = datetime.now()
    try:
        logger.info(f"Received query: {query}")
        # Extract query string from request body
        query_str = query.get("query", "")
        logger.info(f"Processing query: {query_str}")
        
        # Log retriever results
        docs = await retriever.aget_relevant_documents(query_str)
        logger.info(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i + 1} source: {doc.metadata.get('source')}")
            logger.info(f"Document {i + 1} content preview: {doc.page_content[:100]}...")
        
        # Process through chain
        response = await chain.ainvoke(query_str)
        logger.info(f"Generated response: {response}")
        
        return {"answer": response}
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        QUERY_LATENCY.observe((datetime.now() - start_time).total_seconds())

@app.get("/ui")
@app.get("/ui/{path:path}")
async def serve_spa(path: str = ""):
    """Serve the Single Page Application"""
    return FileResponse("frontend/dist/index.html")

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request path: {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    duration = (datetime.now() - start_time).total_seconds()

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
