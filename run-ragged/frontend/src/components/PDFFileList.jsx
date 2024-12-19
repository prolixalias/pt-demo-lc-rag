import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Calendar, 
  Upload, 
  RefreshCw, 
  XCircle, 
  Send, 
  Trash2,
  Menu,
  X
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const PDFFileList = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await fetch('/files');
      const data = await response.json();
      setFiles(data.files);
      setError(null);
    } catch (err) {
      setError('Failed to fetch files. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const deleteFile = async (filename) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }
    
    try {
      setLoading(true);
      const response = await fetch(`/files/${filename}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        await fetchFiles();
        setSidebarOpen(false);
      } else {
        const error = await response.json();
        setError(error.detail || 'Failed to delete file');
      }
    } catch (err) {
      setError('Failed to delete file. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const submitQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    console.log('Submitting query:', query);

    const currentQuery = query.trim();
    setQuery('');
    setQueryLoading(true);
    setSidebarOpen(false);

    setChatHistory(prev => [...prev, { type: 'user', content: currentQuery }]);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: currentQuery }),
      });
      
      const data = await response.json();
      console.log('Response data:', data);
      if (data.answer) {
        setChatHistory(prev => [...prev, { type: 'assistant', content: data.answer }]);
      } else {
        console.error('No answer in response:', data);
        setChatHistory(prev => [...prev, { 
          type: 'error', 
          content: 'Received invalid response format from server' 
        }]);
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { 
        type: 'error', 
        content: 'Failed to get answer. Please try again.' 
      }]);
    } finally {
      setQueryLoading(false);
    }
  };

  const clearChat = () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      setChatHistory([]);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const MessageBubble = ({ message }) => {
    const bubbleClass = {
      user: 'bg-[#E8F5E9] ml-auto border-[#006838] border',
      assistant: 'bg-white border-[#4C230A] border',
      error: 'bg-red-100 border-red-400 border',
    }[message.type];

    return (
      <div 
        className={`max-w-[85%] rounded-lg p-3 mb-3 ${bubbleClass} animate-slideIn border`}
      >
        <ReactMarkdown 
          className="prose max-w-none prose-sm"
          components={{
            pre: ({ children }) => (
              <pre className="bg-[#4C230A] text-white p-2 rounded-lg overflow-x-auto text-sm">
                {children}
              </pre>
            ),
            code: ({ inline, children }) => 
              inline ? (
                <code className="bg-[#E8F5E9] text-[#004D2C] px-1 rounded text-sm">{children}</code>
              ) : (
                children
              ),
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
    );
  };

  if (error) {
    return (
      <div className="text-red-500 p-4 rounded-lg bg-red-50 border border-red-200">
        {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-white pb-6">
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
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold text-[#006838]">PDF Documents</h2>
            <button
              onClick={fetchFiles}
              className="flex items-center gap-2 px-3 py-1.5 bg-[#006838] text-white rounded-lg hover:bg-[#004D2C] transition-colors"
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>

          {files.length === 0 ? (
            <div className="bg-white rounded-lg shadow p-4 border border-[#CFB87C]">
              <div className="flex flex-col items-center justify-center py-8">
                <Upload className="w-8 h-8 text-[#006838] mb-4" />
                <p className="text-[#4C230A] text-base">No PDF files found</p>
                <p className="text-[#4C230A]/70 text-sm mt-2">Upload your first PDF to get started</p>
              </div>
            </div>
          ) : (
            <div className="grid gap-3">
              {files.map((file) => (
                <div key={file.name} className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow border border-[#CFB87C]">
                  <div className="p-3">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-[#E8F5E9] rounded-lg">
                        <FileText className="w-5 h-5 text-[#006838]" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <h3 className="text-base font-semibold text-[#4C230A] truncate">
                            {file.name}
                          </h3>
                          <button
                            onClick={() => deleteFile(file.name)}
                            className="text-red-500 hover:text-red-700 transition-colors p-1"
                            title="Delete file"
                          >
                            <XCircle className="w-4 h-4" />
                          </button>
                        </div>
                        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[#4C230A]/70">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {formatDate(file.created)}
                          </span>
                          <span>{formatFileSize(file.size)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col h-full">
        {/* Chat header */}
        <div className="p-4 border-b border-[#CFB87C] bg-white flex justify-between items-center">
          <h2 className="text-lg font-semibold text-[#006838]">&nbsp;</h2>
          {chatHistory.length > 0 && (
            <button
              onClick={clearChat}
              className="flex items-center gap-2 px-3 py-1.5 text-red-500 hover:text-red-700 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>

        {/* Chat display area */}
        <div className="flex-1 p-4 overflow-y-auto relative">
          {/* Background image */}
          <div 
            className="absolute inset-0 pointer-events-none opacity-10"
            style={{
              backgroundImage: `url("/assets/albear.jpg")`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              backgroundRepeat: 'no-repeat',
            }}
          />
          {/* Chat content */}
          <div className="relative z-10">
            {chatHistory.length === 0 ? (
              <div className="text-center text-[#4C230A]/70 mt-8">
                Ask a question about your documents to get started
              </div>
            ) : (
              chatHistory.map((message, index) => (
                <MessageBubble key={index} message={message} />
              ))
            )}
          </div>
        </div>

        {/* Query input */}
        <div className="p-4 border-t border-[#CFB87C] bg-white">
          <form onSubmit={submitQuery} className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 px-3 py-2 text-sm border border-[#CFB87C] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#006838] text-[#4C230A]"
              disabled={queryLoading}
            />
            <button
              type="submit"
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
  );
};

export default PDFFileList;
