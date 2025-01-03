import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_xai import ChatXAI
from app.debug_metrics import RAGMetrics

logger = logging.getLogger(__name__)

class AIError(Exception):
    pass

class GeminiError(AIError):
    pass

class GrokError(AIError):
    pass

class SynthesisError(AIError):
    pass

class ErrorCategory(Enum):
    INVALID_INPUT = "invalid_input"
    TOKEN_LIMIT = "token_limit"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    CONTEXT_ERROR = "context_error"
    UNKNOWN = "unknown"

@dataclass
class AIResponse:
    content: str
    metadata: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None

class AICollaborationManager:
    def __init__(
        self,
        conversation_memory,
        gemini_llm: Optional[VertexAI] = None,
        grok_api_key: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        debug_mode: bool = False
    ):
        self.conversation_memory = conversation_memory
        self.gemini = gemini_llm
        self.grok_api_key = grok_api_key
        self.prompt_template = prompt_template
        self.debug_mode = debug_mode
        self.grok = ChatXAI(api_key=grok_api_key, model="grok-beta") if grok_api_key else None
        self.metrics = RAGMetrics()

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        error_msg = str(error).lower()
        if "token limit" in error_msg or "sequence too long" in error_msg:
            return ErrorCategory.TOKEN_LIMIT
        elif "rate limit" in error_msg or "quota exceeded" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "invalid request" in error_msg or "invalid input" in error_msg:
            return ErrorCategory.INVALID_INPUT
        elif "context" in error_msg and "error" in error_msg:
            return ErrorCategory.CONTEXT_ERROR
        elif "api" in error_msg and ("error" in error_msg or "failed" in error_msg):
            return ErrorCategory.API_ERROR
        return ErrorCategory.UNKNOWN

    def _create_error_response(self, error: Exception, stage: str, context: Optional[Dict] = None) -> Dict:
        return {
            "type": self._categorize_error(error).value,
            "stage": stage,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "debug_context": context if self.debug_mode and context else None
        }

    async def _get_gemini_response(self, query: str, raw_context: Optional[str] = None) -> AIResponse:
        with self.metrics.track_phase("gemini_generation"):
            try:
                history = await self.conversation_memory.get_context_for_collaboration(num_turns=3)
                
                if raw_context and self.prompt_template:
                    prompt = self.prompt_template.format(
                        context=raw_context,
                        query=query
                    )
                else:
                    prompt = f"""Based on this context and history, respond as Albear Einstein (scientific and helpful) with a JSON object containing 'requires_grok' and 'response' keys.

Previous conversation:
{history}

Current context:
{raw_context or 'No additional context provided'}

Question: {query}

Rules for JSON response:
- Set 'requires_grok' to true for queries needing current/real-time data
- For requires_grok=false, provide full answer in 'response'
- For requires_grok=true, set response=""

JSON Response:"""

                self.metrics.log_debug_info({"stage": "gemini_prompt", "prompt_length": len(prompt), "has_context": bool(raw_context)})
                response = await self.gemini.agenerate([prompt])
                if not response.generations:
                    raise GeminiError("No response generated")

                text = response.generations[0][0].text
                
                try:
                    clean_text = text.strip("```json\n").strip("```")
                    response_obj = json.loads(clean_text)
                    requires_grok = response_obj.get("requires_grok", False)
                    content = response_obj.get("response", "")
                except json.JSONDecodeError:
                    requires_grok = False
                    content = text

                self.metrics.update_token_counts(
                    prompt_tokens=len(prompt) // 4,
                    completion_tokens=len(text) // 4
                )

                return AIResponse(
                    content=content,
                    metadata={
                        "model": "gemini",
                        "success": True,
                        "requires_grok": requires_grok,
                        "source": "rag" if raw_context else "gemini"
                    },
                    debug_info=self.metrics.get_metrics() if self.debug_mode else None
                )

            except Exception as e:
                logger.error(f"Gemini response failed: {str(e)}", exc_info=True)
                self.metrics.log_debug_info({"stage": "gemini_error", "error_type": type(e).__name__, "message": str(e)})
                error_info = self._create_error_response(e, "gemini_generation", 
                    self.metrics.get_metrics() if self.debug_mode else None)
                return AIResponse(
                    content="Error generating response.",
                    metadata={"success": False, "error_type": type(e).__name__},
                    error=error_info,
                    debug_info=self.metrics.get_metrics() if self.debug_mode else None
                )

    async def _get_grok_perspective(self, query: str, gemini_response: str) -> Optional[str]:
        if not self.grok:
            return None

        with self.metrics.track_phase("grok_processing"):
            try:
                prompt = f"""Analyze as Albear Einstein:

Query: {query}
Initial response: {gemini_response}

Provide:
1. Real-time data and updates
2. Any needed corrections
3. Additional context
4. Alternative perspectives"""

                self.metrics.log_debug_info({"stage": "grok_prompt", "prompt_length": len(prompt)})
                response = await self.grok.ainvoke(prompt)
                
                self.metrics.update_token_counts(
                    prompt_tokens=len(prompt) // 4,
                    completion_tokens=len(response.content) // 4
                )

                return response.content

            except Exception as e:
                logger.error(f"Grok processing failed: {str(e)}", exc_info=True)
                self.metrics.log_debug_info({"stage": "grok_error", "error_type": type(e).__name__, "message": str(e)})
                return None

    async def _synthesize_response(self, query: str, gemini_response: str, 
                                 grok_response: Optional[str] = None) -> str:
        with self.metrics.track_phase("response_synthesis"):
            try:
                if not grok_response:
                    return gemini_response

                synthesis_prompt = f"""As Albear Einstein, synthesize:

Query: {query}
Response 1: {gemini_response}
Response 2: {grok_response}

Create a response that:
1. Maintains accuracy
2. Combines unique insights
3. Resolves contradictions
4. Keeps scientific tone

Response:"""

                self.metrics.log_debug_info({
                    "stage": "synthesis_prompt",
                    "prompt_length": len(synthesis_prompt),
                    "has_grok": bool(grok_response)
                })

                response = await self.gemini.agenerate([synthesis_prompt])

                if not response.generations:
                    raise SynthesisError("Synthesis failed")

                final_response = response.generations[0][0].text

                self.metrics.update_token_counts(
                    prompt_tokens=len(synthesis_prompt) // 4,
                    completion_tokens=len(final_response) // 4
                )

                return final_response

            except Exception as e:
                logger.error(f"Response synthesis failed: {str(e)}", exc_info=True)
                self.metrics.log_debug_info({
                    "stage": "synthesis_error", 
                    "error_type": type(e).__name__,
                    "message": str(e)
                })
                return gemini_response

    async def process_query(self, query: str, raw_context: Optional[str] = None,
                          metadata: Dict = None) -> AIResponse:
        self.metrics = RAGMetrics()
        
        with self.metrics.track_phase("query_processing"):
            try:
                logger.info("Starting Gemini response generation")
                gemini_result = await self._get_gemini_response(query, raw_context)

                if raw_context:
                    self.metrics.update_retrieval_stats(
                        chunks=1,
                        scores=[1.0],
                        sources=[metadata.get("source")] if metadata else []
                    )

                grok_response = None
                if self.grok and gemini_result.metadata.get("requires_grok"):
                    logger.info("Starting Grok processing")
                    grok_response = await self._get_grok_perspective(query, gemini_result.content)

                logger.info("Starting response synthesis")
                final_response = await self._synthesize_response(
                    query, gemini_result.content, grok_response
                )

                final_metrics = self.metrics.get_metrics()

                return AIResponse(
                    content=final_response,
                    metadata={
                        **(metadata or {}),
                        "collaboration": {
                            "gemini_used": True,
                            "grok_used": bool(grok_response),
                            "source": gemini_result.metadata.get("source", "unknown"),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    debug_info=final_metrics if self.debug_mode else None
                )

            except Exception as e:
                logger.error(f"Query processing failed: {str(e)}", exc_info=True)
                self.metrics.log_debug_info({
                    "stage": "process_error", 
                    "error_type": type(e).__name__,
                    "message": str(e)
                })
                error_info = self._create_error_response(e, "query_processing",
                    self.metrics.get_metrics() if self.debug_mode else None)
                return AIResponse(
                    content="Error processing query.",
                    metadata={"success": False, "error_type": type(e).__name__},
                    error=error_info,
                    debug_info=self.metrics.get_metrics() if self.debug_mode else None
                )