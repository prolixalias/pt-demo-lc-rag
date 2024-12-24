import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  Calendar,
  Upload,
  RefreshCw,
  XCircle,
  Send,
  Trash2,
  Menu,
  X,
  ThumbsUp,
  ThumbsDown,
  Settings,
  AlertCircle,
  Brain,
  Bug,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { PDFUploadCard } from './components/PDFFileUpload';
import MessageBubble from './components/MessageBubble';
import { FileList } from './components/FileList';
import CollaborationSettings from './components/CollaborationSettings';

// Add new imports
import { 
  useApiRequest, 
  usePersistedState, 
  useDebugMode,
  useChatHistory,
  useCollaborationStatus
} from './hooks';
import { useAICollaboration, useDocumentIndexing } from './hooks/rag';
import { handleError, formatDate, formatFileSize } from './utils';
import { formatModelResponse, processRagDebugInfo } from './utils/rag';

const App = () => {
  // API Request hook
  const { request, loading: apiLoading } = useApiRequest();

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
const [query, setQuery] = useState('');
const [files, setFiles] = useState([]);
const [error, setError] = useState(null);
const [queryLoading, setQueryLoading] = useState(false);
const [sidebarOpen, setSidebarOpen] = useState(false);
const [showWelcomeToast, setShowWelcomeToast] = useState(true);

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
const { status: collaborationStatus } = useCollaborationStatus();
const [settingsOpen, setSettingsOpen] = useState(false);
const [toast, setToast] = useState(null);

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
  try {
    const data = await request('/files');
    setFiles(data.files || []);
    setError(null);
  } catch (err) {
    setError(handleError(err, 'Failed to fetch files'));
    setToast({ 
      message: 'Failed to fetch files', 
      type: 'error' 
    });
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
       
      addMessage({
        type: 'assistant',
        ...response,
        id: Date.now(),
      });

      if (response.error) {
        setToast({
          message: 'Response generated with warnings. Check debug info.',
          type: 'warning'
        });
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
    setToast({ 
      message: 'Reindexing completed successfully', 
      type: 'success' 
    });
  } catch (err) {
    setToast({ 
      message: handleError(err, 'Failed to reindex documents'), 
      type: 'error' 
    });
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

// Side Effects
useEffect(() => {
  fetchFiles();
  if (showWelcomeToast) {
    setToast({ 
      message: 'Ask a question to get started', 
      type: 'info' 
    });
    setShowWelcomeToast(false);
  }
}, []);

// Toast Component
const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000); // Increased to 3 seconds for better visibility
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'success'
    ? 'bg-[#E8F5E9] border-[#006838]'
    : type === 'info'
    ? 'bg-blue-50 border-blue-400'
    : type === 'warning'
    ? 'bg-yellow-50 border-yellow-400'
    : 'bg-red-50 border-red-400';

  const textColor = type === 'success'
    ? 'text-[#006838]'
    : type === 'info'
    ? 'text-blue-700'
    : type === 'warning'
    ? 'text-yellow-700'
    : 'text-red-600';

  return (
    <div 
      className={`
        fixed top-4 left-1/2 transform -translate-x-1/2 z-[60]
        ${bgColor} ${textColor} border rounded-lg px-4 py-2 shadow-lg
        animate-[slideDown_0.3s_ease-out]
        min-w-[200px] text-center
      `}
    >
      {message}
    </div>
  );
};

// Main Render
return (
  <div className="relative min-h-screen">
    {/* Background Image Container */}
    <div
      className="fixed inset-0"
      style={{
        backgroundImage: 'url(/ui/assets/albear.jpg)',
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
                    <span className="animate-pulse">&nbsp;</span>
                  </span>
                )}
              </button>
              
              <button
                onClick={fetchFiles}
                className="flex items-center gap-2 px-3 py-1.5 bg-[#006838] text-white rounded-lg hover:bg-[#004D2C] transition-colors"
                disabled={apiLoading}
              >
                <RefreshCw className={`w-4 h-4 ${apiLoading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>

            <PDFUploadCard onUploadComplete={fetchFiles} />

            <FileList
              files={files}
              onDelete={async (filename) => {
                if (confirm(`Are you sure you want to delete ${filename}?`)) {
                  try {
                    await request(`/files/${filename}`, { method: 'DELETE' });
                    await fetchFiles();
                    setToast({ message: 'File deleted successfully', type: 'success' });
                  } catch (err) {
                    setToast({
                      message: handleError(err, 'Failed to delete file'),
                      type: 'error'
                    });
                  }
                }
              }}
              loading={apiLoading}
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
                onToast={setToast}
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
        onSave={(newSettings) => {
          setCollaborationSettings(newSettings);
          setSettingsOpen(false);
          setToast({ 
            message: 'Settings updated successfully', 
            type: 'success' 
          });
        }}
        onClose={() => setSettingsOpen(false)}
      />
    )}

    {/* Toast Notifications */}
    {toast && (
      <Toast
        message={toast.message}
        type={toast.type}
        onClose={() => setToast(null)}
      />
    )}
  </div>
);
};

export default App;
