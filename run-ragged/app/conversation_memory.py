"""
Conversation History Management Module.

This module provides classes for managing and storing conversation history,
including options for persistence and security filtering. It's designed
to maintain context for AI interactions within a RAG system.

Key Features:
    - Tracks conversation context, source files, and confidence scores.
    - Manages a fixed-size rolling window of conversation history.
    - Supports optional persistence of conversation turns to a vector store.
    - Implements security filtering to remove sensitive information.
    - Provides a method to retrieve relevant history based on user feedback and embeddings.

Classes:
    - ConversationContext: Manages context for a specific interaction
    - ConversationTurn: Represents a single interaction with metadata
    - ConversationMemory: Manages conversation history and data persistence

"""

import json
import logging
import re
from app.logging_config import setup_logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

# Initialize logging with appropriate level
setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Tracks conversation context and source information"""
    raw_context: str
    sanitized_context: str
    source_files: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def to_dict(self) -> Dict:
        """Returns the context as a dictionary."""
        return {
            "sanitized_context": self.sanitized_context,
            "source_files": self.source_files,
            "confidence_score": self.confidence_score
        }

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    query: str
    response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    feedback: Optional[bool] = None
    feedback_weight: float = field(default=1.0)
    metadata: Dict = field(default_factory=dict)
    context: Optional[ConversationContext] = None

    def adjust_weight_from_feedback(self):
        """Adjusts the weight based on user feedback"""
        if self.feedback is True:  # Thumbs up
            self.feedback_weight = 1.5
        elif self.feedback is False:  # Thumbs down
            self.feedback_weight = 0.5
        # If no feedback (None), weight stays at default 1.0

    def to_dict(self) -> Dict:
        """Returns the turn data as a dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "feedback": self.feedback,
            "feedback_weight": self.feedback_weight,
            "context": self.context.to_dict() if self.context else None,
            **self.metadata
        }

class ConversationMemory:
    """
    Manages conversation history with configurable size and persistence.

    Features:
    - Fixed-size rolling window of conversation history
    - Optional persistence to database
    - Metadata tracking for each conversation turn
    - Security filtering for sensitive content
    - Context tracking for AI collaboration
    """

    def __init__(
        self,
        max_turns: int = 10,
        vectorstore: Optional[object] = None,  # PGVector instance if available
        enable_persistence: bool = False
    ):
        """
        Initializes the ConversationMemory object.

        Args:
            max_turns (int): The maximum number of turns to store in history.
            vectorstore (Optional[object]): Optional vectorstore for persistence
            enable_persistence (bool): Whether to enable persistence of turns
        """
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.vectorstore = vectorstore
        self.enable_persistence = enable_persistence

    async def add_interaction(
        self,
        query: str,
        response: str,
        raw_context: Optional[str] = None,
        metadata: Dict = None,
        feedback: Optional[bool] = None
    ) -> None:
        """
        Adds a new interaction with context tracking.

        Args:
            query (str): User's question
            response (str): System's response
            raw_context (Optional[str]): Original RAG context if available
            metadata (Optional[Dict]): Optional metadata about the interaction
            feedback (Optional[bool]): Optional initial feedback
        """
        context = None
        if raw_context:
            sanitized = await self._sanitize_context(raw_context)
            context = ConversationContext(
                raw_context=raw_context,
                sanitized_context=sanitized,
                source_files=metadata.get('source_files', []) if metadata else []
            )

        turn = ConversationTurn(
            query=query,
            response=response,
            metadata=metadata or {},
            feedback=feedback,
            context=context
        )

        self.history.append(turn)

        if self.enable_persistence and self.vectorstore:
            await self._persist_turn(turn)

    async def _sanitize_context(self, raw_context: str) -> str:
        """
        Sanitizes RAG context for safe AI collaboration.

        Implements multi-stage sanitization:
        1. Removes sensitive patterns (emails, IDs, etc.)
        2. Generalizes specific details while preserving meaning
        3. Creates a summary focused on key concepts

        Args:
            raw_context (str): Original RAG context from documents

        Returns:
            str: Sanitized context safe for AI collaboration
        """
        try:
            if not raw_context:
                return ""

            # Stage 1: Remove sensitive patterns
            sanitized = await self._remove_sensitive_patterns(raw_context)

            # Stage 2: Generalize specific details
            sanitized = await self._generalize_content(sanitized)

            # Stage 3: Create focused summary
            sanitized = await self._create_concept_summary(sanitized)

            return sanitized
        except Exception as e:
            logger.error(f"Context sanitization failed: {str(e)}")
            # Return a safe minimal context on error
            return "Context available but details withheld for privacy"

    async def _remove_sensitive_patterns(self, text: str) -> str:
         """Removes potentially sensitive information patterns."""
         # Common sensitive patterns
         patterns = {
             'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
             'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
             'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
             'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
             'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*',
             'path': r'(?:/[a-zA-Z0-9._-]+)+/?',
             'id': r'\b(?:id|ID|Id)\s*[:=]\s*[a-zA-Z0-9_-]+\b'
         }

         sanitized = text
         for pattern_name, pattern in patterns.items():
             # Replace with type indicators
             replacement = f"[{pattern_name.upper()}]"
             sanitized = re.sub(pattern, replacement, sanitized)

         return sanitized

    async def _generalize_content(self, text: str) -> str:
         """Generalizes specific details while preserving meaning."""
         # Patterns for specific details that should be generalized
         generalizations = [
             # Dates to general timeframes
             (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),
             (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),

             # Specific numbers to ranges
             (r'\b\d{6,}\b', '[LARGE_NUMBER]'),
             (r'\$\d+(?:\.\d{2})?', '[AMOUNT]'),

             # Version numbers
             (r'v\d+\.\d+(?:\.\d+)?', '[VERSION]'),

             # File names
             (r'\b[\w-]+\.(pdf|doc|docx|txt)\b', '[DOCUMENT]')
         ]

         generalized = text
         for pattern, replacement in generalizations:
             generalized = re.sub(pattern, replacement, generalized)

         return generalized

    async def _create_concept_summary(self, text: str) -> str:
        """
        Creates a concept-focused summary that preserves key information
        while removing specific implementation details.
        """
        try:
            # Split into sentences (basic approach)
            sentences = [s.strip() for s in text.split('.') if s.strip()]

            # Filter sentences with sensitive keywords
            sensitive_keywords = {
                'password', 'secret', 'key', 'token', 'auth',
                'private', 'credential', 'api', 'config'
            }

            safe_sentences = []
            for sentence in sentences:
                words = set(sentence.lower().split())
                if not words.intersection(sensitive_keywords):
                    safe_sentences.append(sentence)

            # Join remaining sentences
            summary = '. '.join(safe_sentences)

            # If summary is too long, take first and last few sentences
            if len(summary) > 1000:
                parts = safe_sentences[:2]
                if len(safe_sentences) > 4:
                    parts.append('...')
                    parts.extend(safe_sentences[-2:])
                summary = '. '.join(parts)

            return summary if summary else "Content summary not available"

        except Exception as e:
            logger.error(f"Summary creation failed: {str(e)}")
            return "Content summary not available"

    def get_history(
        self,
        turns: int = None,
        format: str = "string",
        include_metadata: bool = False
    ) -> str:
        """
        Retrieves conversation history in specified format.

        Args:
            turns (Optional[int]): Number of recent turns to return (None for all)
            format (str): Output format ("string" or "structured")
            include_metadata (bool): Whether to include turn metadata

        Returns:
            str | List[Dict]: Formatted conversation history
        """
        history_list = list(self.history)
        if turns:
            history_list = history_list[-turns:]

        if format == "string":
            return self._format_history_string(
                history_list,
                include_metadata
            )
        else:
            return self._format_history_structured(
                history_list,
                include_metadata
            )

    def _format_history_string(
        self,
        history_list: List[ConversationTurn],
        include_metadata: bool
    ) -> str:
        """Formats history as a string for LLM context"""
        formatted = []
        for turn in history_list:
            turn_str = f"Human: {turn.query}\nAssistant: {turn.response}"
            if include_metadata and turn.metadata:
                turn_str += f"\nMetadata: {json.dumps(turn.metadata)}"
            if turn.context and turn.context.sanitized_context:
                turn_str += f"\nContext: {turn.context.sanitized_context}"
            formatted.append(turn_str)
        return "\n\n".join(formatted)

    def _format_history_structured(
        self,
        history_list: List[ConversationTurn],
        include_metadata: bool
    ) -> List[Dict]:
        """Formats history as structured data"""
        return [turn.to_dict() for turn in history_list]

    async def _persist_turn(self, turn: ConversationTurn) -> None:
        """Persists a conversation turn with context"""
        try:
            # Create a combined text representation
            text = json.dumps(turn.to_dict())

            # Store in vector database with metadata
            await self.vectorstore.aadd_texts(
                texts=[text],
                metadatas=[{
                    "type": "conversation_turn",
                    "timestamp": turn.timestamp.isoformat(),
                    "has_context": bool(turn.context),
                    **turn.metadata
                }]
            )
        except Exception as e:
            logger.error(f"Failed to persist conversation turn: {str(e)}")

    async def add_feedback(self, turn_index: int, feedback: bool) -> None:
        """
        Adds user feedback to a specific conversation turn

        Args:
            turn_index (int): Index of the turn in history
            feedback (bool): True for thumbs-up, False for thumbs-down
        """
        try:
            if 0 <= turn_index < len(self.history):
                turn = list(self.history)[turn_index]
                turn.feedback = feedback
                turn.adjust_weight_from_feedback()

                # If persistence is enabled, update in vector store
                if self.enable_persistence and self.vectorstore:
                    await self._persist_turn(turn)
                logger.info(f"Feedback added successfully: {feedback} for turn {turn_index}")
                return
            logger.error(f"Turn index out of range: {turn_index}")
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            raise

    def clear(self) -> None:
        """Clears the conversation history"""
        self.history.clear()

    async def get_context_for_collaboration(
        self,
        num_turns: int = 3,
        include_current: bool = True
    ) -> str:
        """
        Gets sanitized conversation context for AI collaboration
        Args:
            num_turns (int): Number of previous turns to include
            include_current (bool): Whether to include current turn
        Returns:
            str: Formatted conversation context safe for sharing
        """
        relevant_turns = list(self.history)[-num_turns:] if include_current else list(self.history)[-(num_turns+1):-1]

        context_parts = []
        for turn in relevant_turns:
            # Add the conversation exchange
            context_parts.append(f"Human: {turn.query}")
            context_parts.append(f"Assistant: {turn.response}")

            # Add sanitized context if available
            if turn.context and turn.context.sanitized_context:
                context_parts.append(f"Context: {turn.context.sanitized_context}")

        return "\n\n".join(context_parts)

    def get_relevant_history(
        self,
        query: str,
        max_turns: int = 3,
        min_weight: float = 0.5
    ) -> str:
        """
        Retrieves conversation history most relevant to current query.
        Useful for maintaining context without overwhelming token limits.

        Args:
            query (str): Current user query
            max_turns (int): Maximum number of relevant turns to return
            min_weight (float): Minimum weight to consider for feedback-weighted results

        Returns:
            str: String of relevant conversation history
        """
        if not self.vectorstore:
            return self.get_history(turns=max_turns)

        try:
            results = self.vectorstore.similarity_search(
                query,
                k=max_turns,
                filter={"type": "conversation_turn"}
            )

            relevant_history = []
            for doc in results:
                try:
                    turn_data = json.loads(doc.page_content)
                    # Only include turns with sufficient weight
                    if turn_data.get('feedback_weight', 1.0) >= min_weight:
                        relevant_history.append(
                            f"Previous relevant exchange:\n"
                            f"Human: {turn_data['query']}\n"
                            f"Assistant: {turn_data['response']}"
                        )
                except json.JSONDecodeError:
                    continue

            return "\n\n".join(relevant_history)

        except Exception as e:
            logger.error(f"Failed to retrieve relevant history: {str(e)}")
            return self.get_history(turns=max_turns)