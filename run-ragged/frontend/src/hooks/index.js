import { useState, useEffect, useCallback, useRef } from 'react';

// Hook for managing API requests with automatic loading and error states
export const useApiRequest = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(async (url, options = {}) => {
    setLoading(true);
    setError(null);

    try {
      // Use the url directly since we're serving from the same origin
      console.log('Making request to:', url); // For debugging

      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Request failed:', errorData); // For debugging
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      console.error('Request error:', err); // For debugging
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { request, loading, error };
};

// Hook for persisting settings in localStorage
export const usePersistedState = (key, initialValue) => {
  const [state, setState] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch (error) {
      console.error('Error writing to localStorage:', error);
    }
  }, [key, state]);

  return [state, setState];
};

// Hook for managing debug state
export const useDebugMode = (collaborationSettings) => {
  const [debugInfo, setDebugInfo] = useState({});

  const debugMode = collaborationSettings?.debug_mode || false;

  const logDebugInfo = useCallback((info) => {
    if (debugMode) {
      setDebugInfo(prev => ({
        ...prev,
        ...info,
        timestamp: new Date().toISOString()
      }));
      console.debug('Debug Info:', info);
    }
  }, [debugMode]);

  return { 
    debugMode,
    debugInfo, 
    logDebugInfo 
  };
};

// Hook for managing file uploads with progress
export const useFileUpload = (uploadUrl, options = {}) => {
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const abortController = useRef(null);

  const upload = useCallback(async (file) => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      abortController.current = new AbortController();

      const xhr = new XMLHttpRequest();

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = (event.loaded / event.total) * 100;
          setProgress(Math.round(percent));
        }
      };

      const response = await new Promise((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status === 200) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.responseText || 'Upload failed'));
          }
        };

        xhr.onerror = () => reject(new Error('Network error'));

        xhr.open('POST', uploadUrl);
        xhr.send(formData);
      });

      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setUploading(false);
      setProgress(0);
      abortController.current = null;
    }
  }, [uploadUrl]);

  const cancelUpload = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
      setUploading(false);
      setProgress(0);
    }
  }, []);

  return {
    upload,
    cancelUpload,
    progress,
    uploading,
    error
  };
};

// Hook for managing chat history with pagination
export const useChatHistory = (pageSize = 20) => {
  const [messages, setMessages] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  const addMessage = useCallback((message) => {
    setMessages(prev => [...prev, {
      ...message,
      id: message.id || Date.now(),
      timestamp: message.timestamp || new Date().toISOString()
    }]);
  }, []);

  const loadMoreMessages = useCallback(() => {
    // Implement loading more messages from backend/storage
    // This is a placeholder for the actual implementation
  }, []);

  const clearHistory = useCallback(() => {
    setMessages([]);
    setCurrentPage(1);
    setHasMore(false);
  }, []);

  return {
    messages,
    addMessage,
    loadMoreMessages,
    clearHistory,
    hasMore,
    currentPage
  };
};

// Hook for managing collaboration status
export const useCollaborationStatus = (pollInterval = 30000) => {
  const [status, setStatus] = useState({
    gemini_available: true,
    grok_available: false,
    memory_enabled: true,
    debug_stats: null
  });

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/debug/collaboration');
      if (response.ok) {
        const data = await response.json();
        setStatus(prev => ({
          ...prev,
          ...data,
          lastUpdated: new Date().toISOString()
        }));
      }
    } catch (error) {
      console.error('Failed to fetch collaboration status:', error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(interval);
  }, [fetchStatus, pollInterval]);

  return {
    status,
    refreshStatus: fetchStatus
  };
};
