"""
Metrics Tracking Implementation Module.

This module provides classes for tracking performance metrics
within the RAG (Retrieval Augmented Generation) pipeline.
It includes detailed timing for each processing phase, tracks
token usage, retrieval statistics, and LLM configuration.

Classes:
    - PhaseMetrics:  Tracks metrics for a specific phase of processing.
    - RAGMetrics: Manages overall metrics for the RAG pipeline.
"""

import logging
import time
from app.logging_config import setup_logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# Initialize logging with appropriate level
setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class PhaseMetrics:
    """Metrics for a specific phase of processing."""
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate phase duration in seconds"""
        if self.end_time is None:
            return 0
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        """Returns the metrics for the phase as a dictionary"""
        return {
            "duration": round(self.duration, 3),
            "success": self.success,
            "error": self.error,
            "details": self.details
        }

class RAGMetrics:
    """Enhanced metrics tracking for RAG pipeline"""
    def __init__(self):
        """Initializes the RAGMetrics object with tracking data"""
        self.phases: Dict[str, PhaseMetrics] = {}
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        # Accumulated metrics
        self.token_counts = {
            "prompt": 0,
            "completion": 0,
            "total": 0
        }
        self.retrieval_stats = {
            "chunks_retrieved": 0,
            "avg_relevance_score": 0.0,
            "sources_used": set()
        }
        self.llm_stats = {
            "model_name": "",
            "temperature": 0.0,
            "max_tokens": 0
        }

    @contextmanager
    def track_phase(self, phase_name: str):
        """Context manager for tracking phase metrics"""
        self.start_phase(phase_name)
        try:
            yield self
        except Exception as e:
            self.end_phase(phase_name, success=False, error=str(e))
            raise
        finally:
            if phase_name in self.phases and self.phases[phase_name].end_time is None:
                self.end_phase(phase_name)

    def start_phase(self, phase_name: str):
        """Start timing a new phase"""
        self.phases[phase_name] = PhaseMetrics(start_time=time.time())
        self.logger.debug(f"Started phase: {phase_name}")

    def end_phase(self, phase_name: str, success: bool = True, error: Optional[str] = None):
        """End timing for a phase"""
        if phase_name in self.phases:
            self.phases[phase_name].end_time = time.time()
            self.phases[phase_name].success = success
            self.phases[phase_name].error = error
            self.logger.debug(f"Ended phase: {phase_name} (success={success})")

    def update_token_counts(self, prompt_tokens: int, completion_tokens: int):
        """Update token usage metrics"""
        self.token_counts["prompt"] += prompt_tokens
        self.token_counts["completion"] += completion_tokens
        self.token_counts["total"] = self.token_counts["prompt"] + self.token_counts["completion"]

    def update_retrieval_stats(self, chunks: int, scores: List[float], sources: List[str]):
        """Update retrieval statistics"""
        self.retrieval_stats["chunks_retrieved"] = chunks
        if scores:
            self.retrieval_stats["avg_relevance_score"] = sum(scores) / len(scores)
        self.retrieval_stats["sources_used"].update(sources)

    def update_llm_stats(self, model_name: str, temperature: float, max_tokens: int):
        """Update LLM configuration stats"""
        self.llm_stats.update({
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        })

    def get_metrics(self) -> Dict:
        """Get complete metrics report"""
        total_duration = time.time() - self.start_time
        
        return {
            "overall": {
                "total_duration": round(total_duration, 3),
                "success": all(phase.success for phase in self.phases.values()),
                "timestamp": datetime.utcnow().isoformat()
            },
            "phases": {
                name: phase.to_dict() 
                for name, phase in self.phases.items()
            },
            "tokens": self.token_counts,
            "retrieval": {
                **self.retrieval_stats,
                "sources_used": list(self.retrieval_stats["sources_used"])
            },
            "llm": self.llm_stats
        }

    def log_debug_info(self, message: str, **kwargs):
        """Log debug information with current context"""
        current_phase = next(
            (name for name, phase in self.phases.items() 
             if phase.end_time is None),
            None
        )
        
        context = {
            "current_phase": current_phase,
            "elapsed_time": time.time() - self.start_time,
            **kwargs
        }
        
        self.logger.debug(f"{message} | Context: {context}")