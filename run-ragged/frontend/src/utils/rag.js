// Source context processing
export const processSourceContext = (context) => {
  if (!context) return [];

  return context.map(source => ({
    id: source.id || `src-${Date.now()}`,
    title: extractTitle(source.content),
    excerpt: extractExcerpt(source.content),
    relevance: source.relevance || 1.0,
    page: source.page || 1,
    timestamp: source.timestamp || new Date().toISOString()
  }));
};

// Extract title from content using heuristics
const extractTitle = (content) => {
  if (!content) return 'Untitled';

  // Try to find a title-like string at the start
  const lines = content.split('\n');
  const firstLine = lines[0].trim();

  // If first line looks like a title, use it
  if (firstLine.length < 100 && !firstLine.endsWith('.')) {
    return firstLine;
  }

  return 'Untitled';
};

// Create a relevant excerpt from content
const extractExcerpt = (content, maxLength = 150) => {
  if (!content) return '';

  // Remove markdown formatting
  content = content.replace(/[#*`]/g, '');

  // Get first sentence or chunk of text
  const excerpt = content.split(/[.!?]/)[0].trim();

  if (excerpt.length <= maxLength) return excerpt;
  return excerpt.substring(0, maxLength) + '...';
};

// Calculate relevance score based on query and content
export const calculateRelevance = (query, content) => {
  if (!query || !content) return 0;

  const queryTerms = query.toLowerCase().split(' ');
  const contentLower = content.toLowerCase();

  // Calculate term frequency
  const termFrequency = queryTerms.reduce((score, term) => {
    const regex = new RegExp(term, 'g');
    const matches = contentLower.match(regex);
    return score + (matches ? matches.length : 0);
  }, 0);

  // Normalize score
  return Math.min(termFrequency / queryTerms.length, 1);
};

// Format model responses for display
export const formatModelResponse = (response, options = {}) => {
  const {
    includeSources = true,
    includeConfidence = true,
    maxSourcesShown = 3
  } = options;

  if (!response) return '';

  let formatted = response.answer || response.content || '';

  if (includeSources && response.sources?.length > 0) {
    const sources = response.sources
      .slice(0, maxSourcesShown)
      .map(src => `[${src.title}](page ${src.page})`)
      .join(', ');

    formatted += `\n\nSources: ${sources}`;
  }

  if (includeConfidence && response.confidence) {
    const confidence = Math.round(response.confidence * 100);
    formatted += `\nConfidence: ${confidence}%`;
  }

  return formatted;
};

// Process debug information for RAG responses
export const processRagDebugInfo = (debugInfo) => {
  if (!debugInfo) return null;

  return {
    retrieval: {
      documentsSearched: debugInfo.docs_searched || 0,
      documentsReturned: debugInfo.docs_returned || 0,
      searchTime: debugInfo.search_time || 0,
      strategy: debugInfo.search_strategy || 'unknown'
    },
    generation: {
      model: debugInfo.model || 'unknown',
      promptTokens: debugInfo.prompt_tokens || 0,
      completionTokens: debugInfo.completion_tokens || 0,
      totalTokens: debugInfo.total_tokens || 0,
      generationTime: debugInfo.generation_time || 0
    },
    performance: {
      totalLatency: debugInfo.total_latency || 0,
      cacheHit: debugInfo.cache_hit || false,
      embeddingTime: debugInfo.embedding_time || 0
    }
  };
};

// Analyze conversation quality
export const analyzeConversation = (messages, options = {}) => {
  const {
    minMessagesForAnalysis = 3,
    feedbackThreshold = 0.6
  } = options;

  if (!messages || messages.length < minMessagesForAnalysis) {
    return { quality: 'insufficient_data' };
  }

  const stats = messages.reduce((acc, msg) => {
    if (msg.feedback) {
      acc.feedbackCount++;
      if (msg.feedback === 'positive') acc.positiveCount++;
    }
    if (msg.error) acc.errorCount++;
    if (msg.sources?.length) acc.withSources++;
    return acc;
  }, {
    feedbackCount: 0,
    positiveCount: 0,
    errorCount: 0,
    withSources: 0
  });

  const feedbackRatio = stats.feedbackCount > 0
    ? stats.positiveCount / stats.feedbackCount
    : 0;

  return {
    quality: feedbackRatio >= feedbackThreshold ? 'good' : 'needs_improvement',
    stats,
    feedbackRatio,
    sourcesRatio: stats.withSources / messages.length,
    errorRate: stats.errorCount / messages.length
  };
};