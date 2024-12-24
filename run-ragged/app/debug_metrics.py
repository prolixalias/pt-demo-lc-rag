from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class RAGMetrics:
    # Content Analysis
    query_length: int = 0
    response_length: int = 0
    num_sources_cited: int = 0
    context_length: int = 0
    document_chunks_used: int = 0
    
    # Vector Search
    similarity_scores: List[float] = None
    nearest_neighbors_distance: float = 0.0
    similarity_threshold_used: float = 0.0
    reranking_applied: bool = False
    chunk_overlap_percentage: float = 0.0
    
    # Token Usage
    prompt_template_tokens: int = 0
    context_tokens: int = 0
    query_tokens: int = 0
    response_tokens: int = 0
    total_cost: float = 0.0
    
    # Memory
    conversation_turns: int = 0
    total_tokens_in_memory: int = 0
    memory_window_size: int = 0
    pruned_messages: int = 0
    
    # Performance Timing
    embedding_generation_time: float = 0.0
    vector_search_time: float = 0.0
    context_processing_time: float = 0.0
    llm_generation_time: float = 0.0
    post_processing_time: float = 0.0
    total_latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to structured dictionary format"""
        raw_dict = asdict(self)
        return {
            "content_analysis": {
                "query_length": raw_dict["query_length"],
                "response_length": raw_dict["response_length"],
                "num_sources_cited": raw_dict["num_sources_cited"],
                "context_length": raw_dict["context_length"],
                "document_chunks_used": raw_dict["document_chunks_used"]
            },
            "vector_search": {
                "similarity_scores": raw_dict["similarity_scores"] or [],
                "nearest_neighbors_distance": raw_dict["nearest_neighbors_distance"],
                "similarity_threshold_used": raw_dict["similarity_threshold_used"],
                "reranking_applied": raw_dict["reranking_applied"],
                "chunk_overlap_percentage": raw_dict["chunk_overlap_percentage"]
            },
            "token_usage": {
                "prompt_template_tokens": raw_dict["prompt_template_tokens"],
                "context_tokens": raw_dict["context_tokens"],
                "query_tokens": raw_dict["query_tokens"],
                "response_tokens": raw_dict["response_tokens"],
                "total_cost": raw_dict["total_cost"]
            },
            "memory": {
                "conversation_turns": raw_dict["conversation_turns"],
                "total_tokens_in_memory": raw_dict["total_tokens_in_memory"],
                "memory_window_size": raw_dict["memory_window_size"],
                "pruned_messages": raw_dict["pruned_messages"]
            },
            "timing": {
                "embedding_generation": raw_dict["embedding_generation_time"],
                "vector_search": raw_dict["vector_search_time"],
                "context_processing": raw_dict["context_processing_time"],
                "llm_generation": raw_dict["llm_generation_time"],
                "post_processing": raw_dict["post_processing_time"],
                "total_latency": raw_dict["total_latency"]
            }
        }

class MetricsTracker:
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = RAGMetrics()
        self._current_phase_start = None

    def start_phase(self, phase_name: str):
        self._current_phase_start = datetime.now()
        return self

    def end_phase(self, phase_name: str):
        if self._current_phase_start:
            duration = (datetime.now() - self._current_phase_start).total_seconds()
            if phase_name == "embedding":
                self.metrics.embedding_generation_time = duration
            elif phase_name == "vector_search":
                self.metrics.vector_search_time = duration
            elif phase_name == "context_processing":
                self.metrics.context_processing_time = duration
            elif phase_name == "llm_generation":
                self.metrics.llm_generation_time = duration
            elif phase_name == "post_processing":
                self.metrics.post_processing_time = duration
        self._current_phase_start = None

    def update_content_metrics(self, query: str, response: str, context: str, docs: List[Any]):
        self.metrics.query_length = len(query)
        self.metrics.response_length = len(response)
        self.metrics.context_length = len(context)
        self.metrics.document_chunks_used = len(docs)
        self.metrics.num_sources_cited = len(set(doc.metadata.get('source') for doc in docs))

    def update_search_metrics(self, similarity_scores: List[float], threshold: float = 0.0):
        self.metrics.similarity_scores = similarity_scores
        self.metrics.similarity_threshold_used = threshold
        if similarity_scores:
            self.metrics.nearest_neighbors_distance = min(similarity_scores)

    def update_token_metrics(self, prompt_tokens: int, context_tokens: int, 
                           query_tokens: int, response_tokens: int, cost_per_token: float = 0.0):
        self.metrics.prompt_template_tokens = prompt_tokens
        self.metrics.context_tokens = context_tokens
        self.metrics.query_tokens = query_tokens
        self.metrics.response_tokens = response_tokens
        self.metrics.total_cost = (prompt_tokens + context_tokens + query_tokens + response_tokens) * cost_per_token

    def update_memory_metrics(self, conversation_history: List[Any], window_size: int, pruned: int = 0):
        self.metrics.conversation_turns = len(conversation_history)
        self.metrics.memory_window_size = window_size
        self.metrics.pruned_messages = pruned
        self.metrics.total_tokens_in_memory = sum(len(msg.get('content', '')) for msg in conversation_history)

    def finalize(self) -> Dict[str, Any]:
        self.metrics.total_latency = (datetime.now() - self.start_time).total_seconds()
        return self.metrics.to_dict()
