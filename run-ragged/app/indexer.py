import os
import asyncio
import tempfile
from google.cloud import storage
from google.cloud.sql.connector import Connector
import pg8000
from langchain_community.vectorstores import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize GCS client
storage_client = storage.Client()

# Initialize database connector
connector = Connector()

def getconn() -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )
    return conn

# Initialize PGVector
store = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args=dict(creator=getconn),
    embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@003"),
    pre_delete_collection=True
)

async def process_pdf(blob_name: str) -> int:
    """Process a single PDF file from GCS bucket."""
    bucket_name = os.getenv("PDF_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("PDF_BUCKET_NAME environment variable is not set")
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        blob.download_to_file(temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        
        # Prepare texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": blob_name,
                "page": chunk.metadata["page"],
                "chunk_index": i
            } for i, chunk in enumerate(chunks)
        ]
        
        # Add to vector store
        ids = store.add_texts(texts, metadatas=metadatas)
        print(f"Processed {blob_name}: {len(ids)} chunks added")
        return len(ids)
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

async def index_pdfs():
    """Index all PDF files in the configured bucket."""
    bucket_name = os.getenv("PDF_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("PDF_BUCKET_NAME environment variable is not set")
    
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in bucket")
        return
    
    tasks = [process_pdf(pdf_file) for pdf_file in pdf_files]
    results = await asyncio.gather(*tasks)
    total_chunks = sum(results)
    print(f"Indexed a total of {total_chunks} chunks from {len(pdf_files)} PDF files")

if __name__ == "__main__":
    asyncio.run(index_pdfs())
