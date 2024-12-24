import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_xai import ChatXAI
from typing import Optional, Dict, Any

# Enhanced logging setup
logger = logging.getLogger(__name__)

class AIError(Exception):
    """Base exception class for AI-related errors"""
    pass

class GeminiError(AIError):
    """Exception for Gemini-specific errors"""
    pass

class GrokError(AIError):
    """Exception for Grok-specific errors"""
    pass

class SynthesisError(AIError):
    """Exception for response synthesis errors"""
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
    """Structured response from AI models"""
    content: str
    metadata: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None

class AICollaborationManager:
    """Manages collaboration between Gemini and Grok with enhanced error handling"""
    
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
        if grok_api_key:
            self.grok = ChatXAI(api_key=grok_api_key, model="grok-beta")
        else:
            self.grok = None

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorizes errors based on their type and message"""
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

    def _create_error_response(
        self,
        error: Exception,
        stage: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """Creates a structured error response"""
        error_category = self._categorize_error(error)
        
        error_info = {
            "type": error_category.value,
            "stage": stage,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.debug_mode and context:
            error_info["debug_context"] = context

        return error_info

    async def _get_gemini_response(
        self,
        query: str,
        raw_context: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None
    ) -> AIResponse:
        """Gets response from Gemini with enhanced error handling"""
        debug_info = {
            "timestamp_start": datetime.utcnow().isoformat(),
            "context_length": len(raw_context) if raw_context else 0
        }

        try:
            history = await self.conversation_memory.get_context_for_collaboration(
                num_turns=3
            )

            # Use the custom prompt template if available
            if raw_context and prompt_template:
                prompt = prompt_template.format(
                    context=raw_context or "No additional context provided",
                    query=query
                )
            else:
                prompt = f"""Based on the following context and conversation history, respond with a JSON object containing keys: 'requires_grok' and 'response'.

Previous conversation:
{history}

Current context:
{raw_context or 'No additional context provided'}

Question: {query}

Here is how you MUST respond:
- You MUST ONLY respond with valid JSON. DO NOT include any other text.
- Set 'requires_grok' to `true` if the query requires access to current information, real-time information, or data not present in the context such as:
  - current weather
  - current news
  - stock prices
  - data from a specific web page
  - etc.
  Otherwise, set `requires_grok` to `false`
- If 'requires_grok' is `false`, set 'response' to a preliminary answer to the query.
- If 'requires_grok' is `true`, set `response` to an empty string ("")

JSON Response:
"""
            debug_info["prompt_length"] = len(prompt)

            logger.info(f"Gemini Prompt: {prompt}")
            response = await self.gemini.agenerate([prompt])
            if not response.generations:
                raise GeminiError("No response generated")
        
            debug_info["timestamp_end"] = datetime.utcnow().isoformat()
        
            generation_chunk = response.generations[0][0]  # Get the first GenerationChunk
            
            # Parse the response as JSON
            try:
                 text_response = generation_chunk.text.strip("```json\n").strip("```")
                 response_obj = json.loads(text_response)
                 if isinstance(response_obj, dict) and "requires_grok" in response_obj and "response" in response_obj:
                    requires_grok = response_obj.get("requires_grok", False)
                    content = response_obj.get("response", "")
                 else:
                    requires_grok = False
                    content = generation_chunk.text
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding json response from Gemini: {str(e)}", exc_info=True)
                requires_grok = False
                content = generation_chunk.text
        
            return AIResponse(
                content=content,
                metadata={"model": "gemini", "success": True, "requires_grok": requires_grok},
                debug_info=debug_info if self.debug_mode else None
            )

        except Exception as e:
            logger.error(f"Gemini response failed: {str(e)}", exc_info=True)
            error_info = self._create_error_response(
                error=e,
                stage="gemini_generation",
                context=debug_info if self.debug_mode else None
            )

            user_message = {
                ErrorCategory.TOKEN_LIMIT: "The input is too long. Please try a shorter question or reduce the context.",
                ErrorCategory.RATE_LIMIT: "The system is currently busy. Please try again in a few moments.",
                ErrorCategory.INVALID_INPUT: "There was an issue with the input format. Please try rephrasing your question.",
                ErrorCategory.CONTEXT_ERROR: "There was an issue processing the context. Please try your question without referring to previous messages.",
                ErrorCategory.API_ERROR: "There was a temporary issue connecting to the AI service. Please try again.",
                ErrorCategory.UNKNOWN: "I encountered an unexpected error. Please try rephrasing your question."
            }[self._categorize_error(e)]

            return AIResponse(
                content=user_message,
                metadata={"model": "gemini", "success": False},
                error=error_info,
                debug_info=debug_info if self.debug_mode else None
            )

    async def _get_grok_perspective(
        self,
        query: str,
        gemini_response: str
    ) -> Optional[str]:
        """Gets perspective from Grok about the query and Gemini's response"""
        if not self.grok:
            return None

        debug_info = {
            "timestamp_start": datetime.utcnow().isoformat()
        }

        try:
            # Format prompt for Grok
            prompt = f"""Analyze this query and the given response. Provide additional insights or corrections if needed.

    Query: {query}

    Gemini's Response: {gemini_response}

    Please analyze the response and provide:
    1. Any factual corrections if needed
    2. Additional relevant context or insights
    3. Alternative perspectives if applicable
    """

            grok_response = await self.grok.ainvoke(prompt)
            grok_response_content = grok_response.content

            debug_info["timestamp_end"] = datetime.utcnow().isoformat()
            debug_info["response_length"] = len(grok_response_content)

            return grok_response

        except Exception as e:
            logger.warning(f"Grok processing failed: {str(e)}")
            return None

    async def _synthesize_response(
        self,
        query: str,
        gemini_response: str,
        grok_response: Optional[str] = None
    ) -> str:
        """Synthesizes a final response combining Gemini and optional Grok input"""
        if not grok_response:
            return gemini_response

        try:
            synthesis_prompt = f"""Given the original query and two AI responses, create a synthesized response that incorporates the best insights from both while maintaining a natural, coherent flow.

Original Query: {query}

Response 1 (Gemini):
{gemini_response}

Response 2 (Grok):
{grok_response}

Create a synthesized response that:
1. Maintains accuracy and factual correctness
2. Combines unique insights from both responses
3. Presents a coherent and natural flow
4. Resolves any contradictions if present

Synthesized Response:"""

            response = await self.gemini.agenerate([synthesis_prompt])

            if not response.generations:
                raise SynthesisError("No response generated during synthesis")

            return response.generations[0][0].text

        except Exception as e:
            logger.error(f"Response synthesis failed: {str(e)}", exc_info=True)
            # On synthesis failure, return Gemini's original response
            return gemini_response

    async def process_query(self, query: str, raw_context: Optional[str] = None, metadata: Dict = None):
        """Processes a query through both AIs collaboratively with enhanced error handling"""
        debug_info = {
            "process_start": datetime.utcnow().isoformat(),
            "query_length": len(query)
        }

        logger.info(f"Starting process_query with query: {query[:100]}...")

        try:
            # Get Gemini's response
            logger.info("Requesting Gemini response")
            gemini_result = await self._get_gemini_response(
                query=query,
                raw_context=raw_context,
                prompt_template=self.prompt_template
            )
            logger.info(f"Gemini response received: {type(gemini_result)}")

            # Get Grok's perspective if available
            grok_response = None
            grok_available = False
            if self.grok and (gemini_result.metadata.get("requires_grok") == True):
                try:
                    grok_response = await self._get_grok_perspective(
                        query=query,
                        gemini_response=gemini_result.content
                    )
                    grok_available = True
                except Exception as e:
                    logger.warning(f"Grok processing failed: {str(e)}")

            # Determine generation method based on actual contribution
            generation_method = "gemini"
            if raw_context:
                generation_method = "rag"
            if grok_response:
                generation_method = "grok"

            # Add actual debug metrics
            debug_info.update({
                "retrieval": {
                    "documentsSearched": len(self.conversation_memory.history) if self.conversation_memory else 0,
                    "documentsReturned": 1 if gemini_result.content else 0,
                    "searchTime": (datetime.utcnow() - datetime.fromisoformat(debug_info["process_start"])).total_seconds(),
                    "strategy": "memory_search"
                },
                "generation": {
                    "model": "gemini",
                    "promptTokens": len(query) // 4,  # Rough approximation
                    "completionTokens": len(gemini_result.content) // 4 if gemini_result.content else 0,
                    "totalTokens": (len(query) + len(gemini_result.content)) // 4 if gemini_result.content else 0,
                    "generationTime": (datetime.utcnow() - datetime.fromisoformat(debug_info["process_start"])).total_seconds()
                },
                "performance": {
                    "totalLatency": (datetime.utcnow() - datetime.fromisoformat(debug_info["process_start"])).total_seconds(),
                    "cacheHit": False,
                    "embeddingTime": 0  # Would need to be measured separately
                },
                "collaboration_status": {
                    "gemini_available": True,
                    "grok_available": grok_available,
                    "memory_enabled": True
                }
            })

            # Synthesize final response
            final_response = await self._synthesize_response(
                query=query,
                gemini_response=gemini_result.content,
                grok_response=grok_response
            )

            # Store the interaction with debug information
            interaction_metadata = {
                **(metadata or {}),
                'collaboration': {
                    'gemini_contributed': True,
                    'grok_contributed': bool(grok_response),
                    'timestamp': datetime.utcnow().isoformat()
                },
                'generation_method': generation_method,
                'debug': debug_info if self.debug_mode else None 
            }

            if self.debug_mode:
                interaction_metadata['debug'] = debug_info

            await self.conversation_memory.add_interaction(
                query=query,
                response=final_response,
                raw_context=raw_context,
                metadata=interaction_metadata
            )

            return AIResponse(
                content=final_response,
                metadata=interaction_metadata,
                error=None,
            )

        except Exception as e:
            logger.error("Query processing failed", exc_info=True)
            error_info = self._create_error_response(
                error=e,
                stage="query_processing",
                context=debug_info if self.debug_mode else None
            )

            logger.error(f"Full error details: {error_info}")

            if self.conversation_memory and self.debug_mode:
                await self.conversation_memory.add_interaction(
                    query=query,
                    response="Error processing query",
                    raw_context=raw_context,
                    metadata={
                        "error": error_info,
                        "debug": debug_info
                    }
                )

            return AIResponse(
                content="I'm having trouble processing your request. Please try again.",
                error=error_info,
                debug_info={
                    **debug_info,
                     "error_details": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
                } if self.debug_mode else None
            )
