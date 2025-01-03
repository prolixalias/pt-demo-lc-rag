import React, { useState, useEffect, useMemo } from 'react';
import { ThumbsUp, ThumbsDown, Bug, ChevronDown, ChevronUp, Database, Cpu, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { processRagDebugInfo } from '../utils/rag';
import { toast } from 'react-toastify';


const DebugPanel = ({ debugInfo, error }) => {
  const processedInfo = useMemo(() => {
    console.log('Processing debug info:', debugInfo);
    if (!debugInfo) return null;

    const retrievalPhase = debugInfo.phases?.retrieval || {};
    const generationPhase = debugInfo.phases?.generation || {};

    return {
      chain_steps: {
        retrieval: {
          time: debugInfo.phases?.retrieval?.duration || 0,
          chunks_retrieved: debugInfo.retrieval?.chunks_retrieved || 0,
          avg_relevance_score: debugInfo.retrieval?.avg_relevance_score || 0,
          sources_used: debugInfo.retrieval?.sources_used || []
        },
        generation: {
          time: debugInfo.phases?.generation?.duration || 0,
          details: debugInfo.phases?.generation?.details || {}
        }
      },
      metrics: {
        total_latency: debugInfo.overall?.total_duration || 0,
        tokens: debugInfo.tokens || {
          prompt: 0,
          completion: 0,
          total: 0
        },
        embedding_time: debugInfo.phases?.embedding?.duration || 0
      },
      model: {
        name: debugInfo.llm?.model_name || 'unknown',
        status: debugInfo.llm || {}
      }
    };
  }, [debugInfo]);

  console.log('Processed debug info:', processedInfo);

  if (!processedInfo && !error) return null;

  return (
    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono overflow-x-auto">
      {error && (
        <div className="mb-2 p-2 bg-red-50 text-red-700 rounded">
          <div><strong>Error Type:</strong> {error.type}</div>
          <div><strong>Message:</strong> {error.message}</div>
        </div>
      )}

      <div className="space-y-2">
        <div>
          <strong>Chain Steps:</strong>
          <div className="ml-2 mt-1">
            <div>Retrieval: {processedInfo.chain_steps.retrieval.time.toFixed(3)}s</div>
            <pre className="text-xs mt-1">{JSON.stringify({
              chunks_retrieved: processedInfo.chain_steps.retrieval.chunks_retrieved,
              avg_relevance_score: processedInfo.chain_steps.retrieval.avg_relevance_score,
              sources_used: processedInfo.chain_steps.retrieval.sources_used
            }, null, 2)}</pre>

            <div className="mt-2">Generation: {processedInfo.chain_steps.generation.time.toFixed(3)}s</div>
            <pre className="text-xs mt-1">{JSON.stringify(processedInfo.chain_steps.generation.details, null, 2)}</pre>
          </div>
        </div>

        <div>
          <strong>Metrics:</strong>
          <pre className="text-xs mt-1">{JSON.stringify({
            total_latency: processedInfo.metrics.total_latency.toFixed(3),
            tokens: processedInfo.metrics.tokens,
            embedding_time: processedInfo.metrics.embedding_time.toFixed(3)
          }, null, 2)}</pre>
        </div>

        <div>
          <strong>Model:</strong>
          <pre className="text-xs mt-1">{JSON.stringify(processedInfo.model, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
};

export const MessageBubble = ({ message, debugMode }) => { // <-- Named export: export const MessageBubble = ...
  const isUser = message.type === 'user';
  const [feedback, setFeedback] = useState(message.feedback);
  const [isExpanded, setIsExpanded] = useState(false);


  useEffect(() => {
    if (!isUser) {
      console.log('Debug Panel Conditions:', {
        debugMode,
        hasMetadataDebug: !!message.metadata?.debug,
        hasDebugInfo: !!message.debug_info,
        debugInfo: message.debug_info,
        metadataDebug: message.metadata?.debug,
        fullMessage: message
      });
    }
  }, [isUser, message, debugMode]);

  const handleFeedback = async (value) => {
    try {
      console.log('Submitting feedback:', {
        message_id: message.id,
        value,
        metadata: message.metadata
      });

      const response = await fetch('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: message.id,
          feedback_value: value,
          source_context: message.metadata?.sources || null,
          metadata: {
            ...message.metadata,
            timestamp: new Date().toISOString()
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to record feedback');
      }

      setFeedback(value);
      if (value === 'positive') {
        toast.success('Thank you for the positive feedback!');
      } else {
        toast.info('Thank you for your feedback');
      }

    } catch (error) {
      console.error('Feedback submission error:', error);
      toast.error('Failed to submit feedback: ' + error.message);

    }
  };

  const messageContent = useMemo(() => {
    console.log('Message content check:', {
      messageType: typeof message,
      answer: message.answer,
      content: message.content,
      response: message.response,
      fullMessage: message
    });

    if (typeof message === 'string') return message;

    return message.answer ||
      message.content ||
      message.response ||
      (message.metadata && message.metadata.content) ||
      '';
  }, [message]);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {/* Message Content */}
      <div
        className={`
          relative
          min-w-[120px]
          max-w-[80%]
          rounded-2xl
          px-4
          pt-4
          pb-2
          shadow-sm
          ${!isUser && 'pt-8'}
          ${isUser ? 'bg-[#E3F2FD] text-gray-800' : 'bg-white/95'}
          ${message.error ? 'border-l-4 border-red-300' : ''}
          ${isUser ? 'rounded-tr-sm' : 'rounded-tl-sm'}
        `}
      >
        {/* Generation Method Badge */}
        {!isUser && (
          <div className="absolute top-2 left-2 flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-gray-100/80 text-xs text-gray-600">
            {message.metadata?.retrieval_stats?.sources && message.metadata.retrieval_stats.sources.length > 0 ? (
              <>
                <Database className="w-3 h-3" />
                <span>RAG</span>
              </>
            ) : message.metadata?.generation_method === 'grok' ? (
              <>
                <Cpu className="w-3 h-3" />
                <span>Grok</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3 h-3" />
                <span>Gemini</span>
              </>
            )}
          </div>
        )}

        {/* Message Content */}
        <div className="prose prose-sm max-w-none mb-2">
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              pre: ({ children }) => (
                <pre className="bg-gray-100 rounded-lg p-3 overflow-x-auto">
                  {children}
                </pre>
              ),
              code: ({ children }) => (
                <code className="bg-gray-100 rounded px-1 py-0.5">
                  {children}
                </code>
              ),
            }}
          >
            {messageContent}
          </ReactMarkdown>
        </div>

        {/* Interactive Footer */}
        {!isUser && (
          <div className="border-t border-gray-200">
            <div className="flex items-start justify-between pt-2"> {/* Changed from items-center to items-start */}
              {/* Left side: Debug Info & Error */}
              <div className="flex flex-col">
                {debugMode && (message.metadata?.debug || message.debug_info) && (
                  <>
                    <button
                      onClick={() => setIsExpanded(!isExpanded)}
                      className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                    >
                      <Bug className="w-3 h-3" />
                      Debug Info
                      {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>

                    {isExpanded && (
                      <DebugPanel
                        debugInfo={message.metadata?.debug || message.debug_info}
                        error={message.error}
                      />
                    )}
                  </>
                )}
                {message.error && (
                  <span className="text-xs text-red-600 flex items-center gap-1">
                    <Bug size={12} />
                    {message.error.type || 'Error'}
                  </span>
                )}
              </div>

              {/* Right side: Feedback & Timestamp */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleFeedback('positive')}
                    className={`p-1.5 rounded-full transition-all duration-200 ${feedback === 'positive'
                        ? 'text-green-600 bg-green-100'
                        : 'text-gray-400 hover:text-green-500 hover:bg-green-50'
                      }`}
                    title="Helpful"
                  >
                    <ThumbsUp className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => handleFeedback('negative')}
                    className={`p-1.5 rounded-full transition-all duration-200 ${feedback === 'negative'
                        ? 'text-red-600 bg-red-100'
                        : 'text-gray-400 hover:text-red-500 hover:bg-red-50'
                      }`}
                    title="Not helpful"
                  >
                    <ThumbsDown className="w-3 h-3" />
                  </button>
                </div>

                <span className="text-xs text-gray-400">
                  {new Date(message.metadata?.collaboration?.timestamp || message.id).toLocaleTimeString(
                    [],
                    {
                      hour: '2-digit',
                      minute: '2-digit'
                    }
                  )}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// export default MessageBubble;