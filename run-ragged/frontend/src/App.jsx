import React, { useState, useEffect, useRef } from 'react';
import { FileText, Calendar, Upload, RefreshCw, XCircle, Send, Trash2, Menu, X, ThumbsUp, ThumbsDown, Settings, AlertCircle, Brain, Bug, ChevronDown, ChevronUp } from 'lucide-react';
import { MessageBubble } from './components/MessageBubble';
import { PDFUploadCard } from './components/PDFFileUpload';
import { FileList } from './components/FileList';
import { CollaborationSettings } from './components/CollaborationSettings';
import { useApiRequest, usePersistedState, useDebugMode, useChatHistory, } from './hooks';
import { useAICollaboration, useDocumentIndexing } from './hooks/rag';
import { handleError, formatDate, formatFileSize } from './utils';
import { formatModelResponse, processRagDebugInfo } from './utils/rag';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const App = () => {
  console.log("App Component Rendered"); // Debug
  // API Request hook
  const { request, loading: apiLoading } = useApiRequest();

  // Solve flickering on indexing check
  const [isFileListRefreshing, setIsFileListRefreshing] = useState(false);

  // Chat History Management
  const {
    messages: chatHistory,
    addMessage,
    clearHistory,
    hasMore,
    loadMoreMessages
  } = useChatHistory(20);

  // Document Management
  const {
    indexDocuments,
    indexingStatus,
    progress: indexingProgress
  } = useDocumentIndexing();

    // State Management
    const [projectName, setProjectName] = useState("");
    const [backgroundImageUrl, setBackgroundImageUrl] = useState(null);
    const [query, setQuery] = useState('');
    const [files, setFiles] = useState([]);
    const [error, setError] = useState(null);
    const [queryLoading, setQueryLoading] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [showWelcomeToast, setShowWelcomeToast] = useState(true);
    const [isIndexing, setIsIndexing] = useState(false); // <-- ADDED: State for indexing status

    // Persisted Settings
    const [collaborationSettings, setCollaborationSettings] = usePersistedState('collaboration_settings', {
      enable_grok: true,
      max_conversation_turns: 3,
      synthesis_temperature: 0.3,
      debug_mode: false,
      grok_temperature: 0.7,
      gemini_temperature: 0.3
    });

    // Debug and Status Management
    const { debugMode, debugInfo, logDebugInfo } = useDebugMode(collaborationSettings);
    const [settingsOpen, setSettingsOpen] = useState(false);

    // AI Collaboration
    const { getModelResponse, modelStatus } = useAICollaboration(collaborationSettings);

    // Refs
    const chatContainerRef = useRef(null);

    // Utility Functions
    const scrollToBottom = () => {
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      }
    };

    const fetchFiles = async () => {
      setIsFileListRefreshing(true); // <-- Set loading to true when fetch starts
      try {
        const data = await request('/files');
        setFiles(data.files || []);
        setError(null);
      } catch (err) {
        setError(handleError(err, 'Failed to fetch files'));
        toast.error('Failed to fetch files');
      } finally {
        setIsFileListRefreshing(false); // <-- Set loading to false when fetch completes (or errors)
      }
    };

  // Main Handlers
  const handleQuerySubmit = async (e) => {
    try {
      // Make sure we prevent default before anything else
      if (e && e.preventDefault) {
        e.preventDefault();
      }

      console.log('Form submission starting...');

      if (!query || !query.trim()) {
        console.log('Empty query, skipping submission');
        return;
      }

      const currentQuery = query.trim();
      console.log('Processing query:', currentQuery);

      // Clear form and set loading state
      setQuery('');
      setQueryLoading(true);
      setSidebarOpen(false);
      setShowWelcomeToast(false);

      // Add user message
      addMessage({
        type: 'user',
        content: currentQuery,
        id: Date.now()
      });

      try {
        // Get response using AI collaboration hook
        const response = await getModelResponse(currentQuery);
        console.log('Got response:', response);
        console.log('Debug info:', response.debug_info);

        addMessage({
          type: 'assistant',
          ...response,
          id: Date.now(),
          debug_info: response.debug_info || response.metadata?.debug
        });

        if (response.error) {
          toast.warning('Response generated with warnings. Check debug info.');
        }
      } catch (error) {
        console.error('Error during query processing:', error);
        const errorMessage = handleError(error, 'Failed to process query');
        addMessage({
          type: 'assistant',
          content: `Error: ${errorMessage}. Please try again.`,
          error: {
            type: 'query_error',
            message: errorMessage,
            timestamp: new Date().toISOString()
          },
          debugInfo: collaborationSettings.debug_mode ? {
            error_type: error.name,
            error_stack: error.stack,
            query: currentQuery
          } : null
        });
      }
    } catch (outerError) {
      console.error('Outer error in form submission:', outerError);
    } finally {
      setQueryLoading(false);
      setTimeout(scrollToBottom, 100);
    }
  };

  const handleReindex = async () => {
    try {
      await indexDocuments();
      await fetchFiles();
      toast.success('Reindexing completed successfully');
      setTimeout(() => {
        setSidebarOpen(false);
      }, 100);
    } catch (err) {
      toast.error(handleError(err, 'Failed to reindex documents'));
    }
  };

  const handleClearHistory = () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      clearHistory();
    }
  };

  const handleScroll = (e) => {
    const { scrollTop } = e.target;
    if (scrollTop === 0 && hasMore) {
      loadMoreMessages();
    }
  };

  // New function to fetch indexing status
  const fetchIndexingStatus = async () => { // <-- ADDED: fetchIndexingStatus function
    try {
      const data = await request('/indexer/status');
      setIsIndexing(data.is_indexing);
    } catch (error) {
      console.error("Failed to fetch indexing status:", error);
      setIsIndexing(false); // Set to false in case of error
    }
  };

  // Fetch health status with project name on mount, fetch background image
  useEffect(() => {
    const fetchHealthStatus = async () => {
      try {
        const data = await request('/health');
          if (data && data.project_name) {
              setProjectName(data.project_name);

              // Check for background image by attempting to load it
              const imageUrl = `/ui/assets/${data.project_name.toLowerCase()}.jpg`;

              const image = new Image();
              image.onload = () => {
                  setBackgroundImageUrl(imageUrl);
              };
              image.onerror = () => {
                  setBackgroundImageUrl(null); // Default
              };
              image.src = imageUrl;
          }
        else {
          // Handle the case where project_name is missing
          console.warn("Project name not found in /health response");
          setProjectName('default'); // Fallback default
          setBackgroundImageUrl(null); // Set background image to null
        }
      } catch (err) {
        console.error("Failed to fetch health status:", err);
          setProjectName('default'); // Fallback default
          setBackgroundImageUrl(null); // Set background image to null
      }
    };
      fetchHealthStatus();
  }, [request]);

  // Fetch indexing status periodically
  useEffect(() => { // <-- ADDED: useEffect for periodic indexing status fetch
    fetchIndexingStatus(); // Fetch status immediately

    const intervalId = setInterval(fetchIndexingStatus, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [request]);

    // Side Effects
    useEffect(() => {
      fetchFiles();
      toast.info('Ask a question to get started');
      if (showWelcomeToast) {
        setShowWelcomeToast(false);
      }
    }, []);

    const memoizedFiles = React.useMemo(() => files, [files]);

    // Main Render
    return (
      <div className="relative min-h-screen">
      <style jsx>{`
        @keyframes pulse-slow {
          0% { opacity: 1; }
          50% { opacity: 0.4; }
          100% { opacity: 1; }
        }

        .animate-pulse-slow {
          animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
      `}</style>
        {/* Background Image Container */}
        <div
          className="fixed inset-0"
          style={{
           backgroundImage: backgroundImageUrl ? `url(${backgroundImageUrl})` : 'none',
            backgroundColor: backgroundImageUrl ? 'transparent' : 'white',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
            zIndex: -1
          }}
        />

        {/* Main Content */}
        <div className="relative min-h-screen z-10 bg-white/30">
          <div className="flex flex-col h-screen">

            {/* Menu Button */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="fixed top-4 left-4 z-50 p-2 bg-[#006838] text-white rounded-lg shadow-lg"
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>

            {/* Sidebar */}
            <div
              className={`
              fixed inset-0 z-40
              w-3/4 min-[1024px]:w-1/4 max-[1023px]:w-3/4
              p-4
              bg-white
              transform transition-transform duration-300 ease-in-out
              ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
              overflow-y-auto
              pt-16
            `}
            >
              {/* Settings Button */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="absolute top-4 right-4 p-2 text-[#4C230A] hover:text-[#006838] transition-colors"
                title="Settings"
              >
                <Settings size={20} />
              </button>

              {/* File Management UI */}
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2"> {/* Container for button and LED */}
                    <button
                      onClick={handleReindex}
                      className="flex items-center gap-2 px-3 py-1.5 bg-[#CFB87C] text-white rounded-lg hover:bg-[#B39B59] transition-colors"
                      disabled={indexingStatus === 'indexing'}
                    >
                      <RefreshCw
                        className={`w-4 h-4 ${indexingStatus === 'indexing' ? 'animate-spin' : ''}`}
                      />
                      <span>{indexingStatus === 'indexing' ? 'Indexing...' : 'Reindex'}</span>
                      {indexingStatus === 'indexing' && (
                        <span className="text-xs opacity-75">
                          <span className="animate-pulse"> </span>
                        </span>
                      )}
                    </button>
                    {/* LED Indicator */}
                    <div
                      className={`w-3 h-3 rounded-full border border-gray-400 ml-2 ${
                        isIndexing ? 'bg-green-500 animate-pulse-slow border-green-700' : 'bg-gray-300'
                      }`}
                      title={isIndexing ? "Indexing in progress" : "Indexing idle"}
                    />
                  </div>

      
                  <button
                    onClick={fetchFiles}
                    className="flex items-center gap-2 px-3 py-1.5 bg-[#006838] text-white rounded-lg hover:bg-[#004D2C] transition-colors"
                    disabled={apiLoading} // Keep disabled prop as-is (based on apiLoading)
                  >
                    <RefreshCw className={`w-4 h-4 ${isFileListRefreshing ? 'animate-spin' : ''}`} /> {/* <-- Use isFileListRefreshing for animation */}
                    <span>Refresh</span>
                  </button>    
                </div>

                <PDFUploadCard onUploadComplete={fetchFiles} />

                <FileList
                  files={memoizedFiles} // <-- Pass the memoized files prop
                  onDelete={async (filename) => {
                    if (confirm(`Are you sure you want to delete ${filename}?`)) {
                      try {
                        await request(`/files/${filename}`, { method: 'DELETE' });
                        await fetchFiles();
                        toast.success('File deleted successfully');
                      } catch (err) {
                        toast.error(handleError(err, 'Failed to delete file'));
                      }
                    }
                  }}
                  // loading={apiLoading}
                  error={error}
                />
              </div>
            </div>

            {/* Sidebar Overlay */}
            {sidebarOpen && (
              <div
                className="fixed inset-0 bg-black/20 z-30"
                onClick={() => setSidebarOpen(false)}
              />
            )}

            {/* Chat Messages with updated scroll handler */}
            <div
              ref={chatContainerRef}
              className="flex-1 p-4 overflow-y-auto"
              onScroll={handleScroll}
            >
              <div className="space-y-4">
                {chatHistory.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    debugMode={debugMode}
                  />
                ))}
              </div>
            </div>

            {/* Query Input */}
            <div className="p-4 bg-white/80">
              <form
                onSubmit={handleQuerySubmit}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a question..."
                  className="flex-1 px-3 py-2 text-sm border border-[#CFB87C] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#006838]"
                  disabled={queryLoading}
                />
                <button
                  type="submit"
                  onClick={(e) => {
                    // Backup prevention of form submission
                    e.preventDefault();
                    handleQuerySubmit(e);
                  }}
                  className="flex items-center gap-2 px-4 py-2 bg-[#006838] text-white rounded-lg hover:bg-[#004D2C] transition-colors disabled:bg-[#006838]/50"
                  disabled={queryLoading || !query.trim()}
                >
                  {queryLoading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                  Send
                </button>
              </form>
            </div>
          </div>
        </div>
        {/* Settings Modal */}
        {settingsOpen && (
          <CollaborationSettings
            initialSettings={collaborationSettings}
            onSave={(newSettings, onToastCallback) => {
              setCollaborationSettings(newSettings);
              if (onToastCallback) {
                onToastCallback('Settings updated successfully');
              }
            }}
            onClose={() => setSettingsOpen(false)}
          />
        )}
        <ToastContainer position="top-center" autoClose={3000} />
      </div>
    );
  };

  export default App;