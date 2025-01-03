import { useState, useCallback, useEffect } from 'react';
import { useApiRequest } from './index';

// Hook for managing source context and citations
export const useSourceContext = () => {
  const [sources, setSources] = useState([]);
  const [activeSource, setActiveSource] = useState(null);

  const addSource = useCallback((source) => {
    setSources(prev => {
      const exists = prev.some(s => s.id === source.id);
      if (exists) return prev;
      return [...prev, source];
    });
  }, []);

  const trackSourceUsage = useCallback((sourceId) => {
    setSources(prev => prev.map(source =>
      source.id === sourceId
        ? { ...source, usageCount: (source.usageCount || 0) + 1 }
        : source
    ));
  }, []);

  return {
    sources,
    activeSource,
    setActiveSource,
    addSource,
    trackSourceUsage
  };
};

// Hook for managing AI model interactions
export const useAICollaboration = (settings = {}) => {
  const { request } = useApiRequest();
  const [modelStatus, setModelStatus] = useState({
    gemini: 'idle',
    grok: 'idle'
  });

  const getModelResponse = useCallback(async (query, context) => {
    const payload = {
      query,
      context,
      collaboration_settings: settings
    };
    setModelStatus(prev => ({ ...prev, gemini: 'loading' }));
    try {
      const response = await request('/query', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      console.log('Raw API response:', response);
      setModelStatus(prev => ({ ...prev, gemini: 'idle' }));
      return response;
    } catch (error) {
      setModelStatus(prev => ({ ...prev, gemini: 'error' }));
      throw error;
    }
  }, [request, settings]);

  return {
    getModelResponse,
    modelStatus
  };
};

// Hook for managing conversation feedback
export const useConversationFeedback = () => {
  const { request } = useApiRequest();
  const [feedbackStats, setFeedbackStats] = useState({
    positive: 0,
    negative: 0,
    total: 0
  });

  const submitFeedback = useCallback(async (messageId, value, context = {}) => {
    await request('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        message_id: messageId,
        feedback_value: value,
        context
      })
    });
    setFeedbackStats(prev => ({
      ...prev,
      [value]: prev[value] + 1,
      total: prev.total + 1
    }));
  }, [request]);

  const getFeedbackStats = useCallback(async () => {
    const response = await request('/feedback/analytics');
    setFeedbackStats(response);
    return response;
  }, [request]);

  return {
    submitFeedback,
    getFeedbackStats,
    feedbackStats
  };
};

// Hook for managing document indexing
export const useDocumentIndexing = () => {
  const { request } = useApiRequest();
  const [indexingStatus, setIndexingStatus] = useState('idle');
  const [progress, setProgress] = useState(0);

  const indexDocuments = useCallback(async (files = null) => {
    setIndexingStatus('indexing');
    setProgress(0);
    try {
      console.log('Starting document indexing...');

      const response = await request('/files/reindex', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('Indexing response:', response);
      setIndexingStatus('completed');
      setProgress(100);
      return response;
    } catch (error) {
      console.error('Indexing failed:', error);
      setIndexingStatus('error');
      setProgress(0);
      throw error;
    }
  }, [request]);

  return {
    indexDocuments,
    indexingStatus,
    progress
  };
};
