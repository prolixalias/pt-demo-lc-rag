# run-ragged

## langchain flow diagram
```mermaid
flowchart TD
    %% User Input and Initial Processing
    A[User Question] --> B[Prompt]
    B --> C{Check Local Knowledge}
    
    %% Knowledge Source Decision Tree
    C -->|Found| D[Generate RAG Response]
    C -->|Not Found| E{Needs Real-time Data?}
    
    %% External API Paths
    E -->|No| F[Query Gemini API]
    E -->|Yes| G[Query Grok API]
    
    %% Response Processing
    D --> H[Format in Prompt]
    F --> H
    G --> H
    
    %% Output and Feedback Loop
    H --> I[Response to User]
    J[User Feedback] --> K[Store for Training]
    
    %% Configuration Management
    L[Local Config] --> B
    L --> C
    L --> E

    %% Debug Information Flow
    style M fill:#f9f,stroke:#333,stroke-width:2px
    M[Debug Info Panel] -.-> B
    M -.-> C
    M -.-> D
    M -.-> F
    M -.-> G
    M -.-> H

    %% Collaboration Status
    N[Hamburger Menu] -.-> O[Status Report]
    
    %% Style Definitions
    classDef api fill:#f96,stroke:#333,stroke-width:2px
    classDef config fill:#9cf,stroke:#333,stroke-width:2px
    classDef process fill:#9f9,stroke:#333,stroke-width:2px
    
    %% Apply Styles
    class F,G api
    class L config
    class B,D,H process
```
## python class diagram
```mermaid
classDiagram
    class AICollaborationManager {
        -ConversationMemory memory
        -VertexAI gemini
        -ChatXAI grok
        -PromptTemplate template
        +process_query()
        -_get_gemini_response()
        -_get_grok_perspective()
        -_synthesize_response()
    }
    
    class ConversationMemory {
        -deque history
        -PGVector vectorstore
        +add_interaction()
        +get_history()
        +get_context_for_collaboration()
        -_sanitize_context()
    }
    
    class FastAPIServer {
        +query_documents()
        +upload_pdf()
        +list_files()
        +reindex_documents()
    }
    
    class IndexerService {
        +process_pdf()
        +index_pdfs()
        +add_embeddings()
    }
    
    class MetricsTracker {
        +start_phase()
        +end_phase()
        +update_search_metrics()
        +finalize()
    }

    AICollaborationManager --> ConversationMemory
    FastAPIServer --> AICollaborationManager
    FastAPIServer --> IndexerService
    IndexerService --> ConversationMemory
    AICollaborationManager --> MetricsTracker
```
## Query path flow diagram
```mermaid
flowchart TD
    A[Query] --> B{Contains Keywords?}
    B -->|Weather, Sports, News| C[Realtime/Grok]
    B -->|No Keywords| D{Vector Search}
    D -->|Results Found| E[RAG Response]
    D -->|No Results| F[Gemini Response]
    
    style C fill:#f96,stroke:#333
    style E fill:#9f9,stroke:#333
```
## Query processing sequence
```mermaid
sequenceDiagram
    participant U as User
    participant P as QueryProcessor
    participant V as VectorStore
    participant G as Gemini
    participant K as Grok
    
    U->>P: Query
    
    %% RAG Attempt First
    P->>V: Search for Context
    V-->>P: Return Documents
    
    alt Documents Found
        P->>G: Query + RAG Context
        G-->>P: Enhanced Response
        P->>U: Return RAG Response
    else No Documents Found
        P->>G: Query (No Context)
        G-->>P: Response with requires_grok flag
        
        alt requires_grok = true
            P->>K: Get Realtime Data
            K-->>P: Realtime Data
            P->>G: Synthesize Final Response
            G-->>P: Final Response
            P->>U: Return Synthesized Response
        else requires_grok = false
            P->>U: Return Direct Gemini Response
        end
    end

    note over P: Always try RAG first,<br/>only use realtime if no<br/>RAG results and Gemini<br/>flags realtime needed
```
## Debug flow diagram
```mermaid
flowchart TD
    A[User Query] --> B[Log Query Type Detection]
    B --> C{Document Search}
    C -->|Success| D[Log Retrieved Chunks]
    C -->|Failure| E[Log Search Error]
    D --> F[Log RAG Integration]
    E --> G[Log Fallback Path]
    F --> H{API Selection}
    G --> H
    H -->|Gemini| I[Log Gemini Response]
    H -->|Grok| J[Log Grok Response]
    I & J --> K[Log Final Response]
    K --> L[Log UI Updates]
    
    style D fill:#f96
    style E fill:#f96
    style I fill:#f96
    style J fill:#f96
```
### Flow debug points
```mermaid
sequenceDiagram
    participant C as Client
    participant F as FastAPI
    participant V as VectorStore
    participant AI as AICollaboration
    participant G as Gemini

    C->>F: Query Request
    F->>V: similarity_search()
    Note over F,V: First Potential Break Point
    
    alt Vector Search Success
        V-->>F: Retrieved Documents
        F->>AI: Process with RAG Context
        AI->>G: Generate with Context
        G-->>AI: Enhanced Response
        AI-->>F: Final Response
    else Vector Search Failure
        V-->>F: No Results/Error
        Note over F,V: Second Break Point
        F->>AI: Process without Context
        AI->>G: Direct Generation
        G-->>AI: Base Response
        AI-->>F: Fallback Response
    end
    
    F-->>C: Response to Client
```
## Monitoring points
```mermaid
flowchart TD
    A[Client Request] --> B[Load Balancer/Health Check]
    B --> C[FastAPI Server]
    
    subgraph Critical Monitoring Points
    C --> D{Vector Search}
    D --> |Success| E[Document Processing]
    D --> |Failure| F[Fallback Path]
    
    E --> G{API Calls}
    F --> G
    
    G --> |Basic| H[Gemini]
    G --> |Real-time| I[Grok -> Gemini]
    end
    
    H --> J[Response]
    I --> J
    
    style B fill:#ff9999,stroke:#333,stroke-width:2px
    style D fill:#ff9999,stroke:#333,stroke-width:2px
    style G fill:#ff9999,stroke:#333,stroke-width:2px
    
    %% Monitoring metrics
    M1[Response Time] -.-> B
    M2[Memory Usage] -.-> D
    M3[API Latency] -.-> G
    
    classDef metric fill:#ccffcc,stroke:#333,stroke-width:1px
    class M1,M2,M3 metric
```

## global
```shell
eval "$(devbox global shellenv)"
```
## development packages
```shell
devbox shell
```
## secrets / config
```shell
doppler setup
```
## environment variables
```shell
export PROJECT_PREFIX='pt'
export PROJECT_NAME='demo-lc-rag'
export PROJECT_SUFFIX=$(date +%Y%m%d)
export PROJECT=${PROJECT_PREFIX}-${PROJECT_NAME}-${PROJECT_SUFFIX}
```
## cloud setup
```shell
gcloud auth print-access-token > /dev/null 2>&1 || gcloud auth login
gcloud projects describe ${PROJECT} > /dev/null 2>&1 || gcloud projects create ${PROJECT}
gcloud projects list --filter="name:${PROJECT} AND lifecycleState:ACTIVE" || gcloud config set project ${PROJECT}
gcloud auth application-default print-access-token > /dev/null 2>&1 || gcloud auth --quiet application-default login
gcloud services enable cloudresourcemanager.googleapis.com --project=${PROJECT}
gcloud auth application-default set-quota-project ${PROJECT} && doppler run --command="gcloud billing projects link ${PROJECT} --billing-account=\${GCLOUD_BILLING_FFI}"
gcloud projects list --filter="name:${PROJECT} AND lifecycleState:ACTIVE" && gcloud config set project ${PROJECT}
gcloud config get project | grep -q ${PROJECT} && gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  bigqueryconnection.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  documentai.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com
```
## add database
```shell
doppler run --command="gcloud sql instances describe \${DB_INSTANCE_DEMO} > /dev/null 2>&1 || gcloud sql instances create \${DB_INSTANCE_DEMO} \
  --database-version \${DB_VERSION_DEMO} \
  --tier \${DB_TIER_DEMO} \
  --region \${GCLOUD_REGION}"
doppler run --command="gcloud sql databases describe \${DB_NAME_DEMO} --instance=\${DB_INSTANCE_DEMO} > /dev/null 2>&1 || gcloud sql databases create \${DB_NAME_DEMO} \
  --instance \${DB_INSTANCE_DEMO}"
doppler run --command="gcloud sql users describe \${DB_USERNAME_DEMO} --instance=\${DB_INSTANCE_DEMO} > /dev/null 2>&1 || gcloud sql users create \${DB_USERNAME_DEMO} \
  --instance \${DB_INSTANCE_DEMO} \
  --password \${DB_PASSWORD_DEMO}"
```
## add storage
```shell
doppler run --command="gcloud storage buckets get-iam-policy gs://\${PDF_BUCKET_DEMO}-\${PROJECT_SUFFIX} > /dev/null 2>&1 || gcloud storage buckets create gs://\${PDF_BUCKET_DEMO}-\${PROJECT_SUFFIX} --project=\${PROJECT} --location=\${GCLOUD_REGION}"
```
## build+run indexer
```shell
doppler run --command="gcloud run deploy indexer \
  --quiet \
  --timeout=60m \
  --memory 4G \
  --source run-ragged/. \
  --command python \
  --args /code/app/indexer.py \
  --set-env-vars=DB_INSTANCE_NAME=\$(gcloud sql instances describe \${DB_INSTANCE_DEMO} --format 'value(connectionName)') \
  --set-env-vars=DB_NAME=\${DB_NAME_DEMO} \
  --set-env-vars=DB_PASS=\${DB_PASSWORD_DEMO} \
  --set-env-vars=DB_USER=\${DB_USERNAME_DEMO} \
  --set-env-vars=PDF_BUCKET_NAME=\${PDF_BUCKET_DEMO}-${PROJECT_SUFFIX} \
  --set-env-vars=PRE_SHARED_KEY=\${PRE_SHARED_KEY} \
  --set-env-vars=SERVICE_TYPE=indexer \
  --set-env-vars=LOG_LEVEL_INDEXER=\$LOG_LEVEL_INDEXER \
  --set-env-vars=MAX_FILE_SIZE=\$MAX_FILE_SIZE \
  --set-env-vars=MAX_CHUNKS=\$MAX_CHUNKS \
  --set-env-vars=CHUNK_SIZE=\$CHUNK_SIZE \
  --set-env-vars=CHUNK_OVERLAP=\$CHUNK_OVERLAP \
  --set-env-vars=EMBEDDING_BATCH_SIZE=\$EMBEDDING_BATCH_SIZE \
  --region=\${GCLOUD_REGION} \
  --allow-unauthenticated"
```
## build+run API
```shell
doppler run --command="gcloud run deploy run-ragged \
  --quiet \
  --timeout=10m \
  --memory 1G \
  --source run-ragged/. \
  --set-env-vars=CIRCUIT_BREAKER_FAILURE_THRESHOLD=\$CIRCUIT_BREAKER_FAILURE_THRESHOLD \
  --set-env-vars=CIRCUIT_BREAKER_RECOVERY_TIMEOUT=\$CIRCUIT_BREAKER_RECOVERY_TIMEOUT \
  --set-env-vars=DB_INSTANCE_NAME=\$(gcloud sql instances describe \${DB_INSTANCE_DEMO} --format 'value(connectionName)') \
  --set-env-vars=DB_NAME=\${DB_NAME_DEMO} \
  --set-env-vars=DB_PASS=\${DB_PASSWORD_DEMO} \
  --set-env-vars=DB_USER=\${DB_USERNAME_DEMO} \
  --set-env-vars=GROK_API_KEY=\$GROK_API_KEY \
  --set-env-vars=INDEXER_HEALTH_CHECK_INTERVAL=\$INDEXER_HEALTH_CHECK_INTERVAL \
  --set-env-vars=INDEXER_SERVICE_URL=\$(gcloud run services describe indexer --region \${GCLOUD_REGION} --project ${PROJECT} --format='get(metadata.annotations.\"run.googleapis.com/urls\")' | jq -r '.[0]') \
  --set-env-vars=LLM_MAX_OUTPUT_TOKENS=\$LLM_MAX_OUTPUT_TOKENS \
  --set-env-vars=LLM_MODEL_NAME=\$LLM_MODEL_NAME \
  --set-env-vars=LLM_TEMPERATURE=\$LLM_TEMPERATURE \
  --set-env-vars=LOG_LEVEL_API=\$LOG_LEVEL_API \
  --set-env-vars=PDF_BUCKET_NAME=pdf-bucket-${PROJECT_SUFFIX} \
  --set-env-vars=PRE_SHARED_KEY=\${PRE_SHARED_KEY} \
  --set-env-vars=SERVICE_TYPE=server \
  --region=\$GCLOUD_REGION \
  --allow-unauthenticated"
```
