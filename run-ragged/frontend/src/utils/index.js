// File size formatting
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

// Date formatting
export const formatDate = (date, options = {}) => {
  const defaultOptions = {
    includeTime: true,
    relative: false
  };

  const opts = { ...defaultOptions, ...options };
  const d = new Date(date);

  if (opts.relative) {
    const now = new Date();
    const diff = now - d;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
  }

  const dateStr = d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });

  if (opts.includeTime) {
    const timeStr = d.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
    return `${dateStr} ${timeStr}`;
  }

  return dateStr;
};

// Error handling
export const handleError = (error, defaultMessage = 'An error occurred') => {
  if (error.response) {
    try {
      return error.response.data.detail || error.response.data.message || defaultMessage;
    } catch (e) {
      return defaultMessage;
    }
  }
  return error.message || defaultMessage;
};

// Debug information formatting
export const formatDebugInfo = (info) => {
  if (!info) return null;

  return {
    timing: {
      start: info.process_start,
      end: info.process_end,
      duration: info.duration
    },
    memory: {
      used: formatFileSize(info.memory_used || 0),
      peak: formatFileSize(info.memory_peak || 0)
    },
    model: {
      name: info.model_name,
      version: info.model_version,
      temperature: info.temperature
    },
    tokens: {
      input: info.input_tokens,
      output: info.output_tokens,
      total: info.total_tokens
    }
  };
};

// Collaboration status helpers
export const getCollaborationStatusSummary = (status) => {
  if (!status) return 'Unknown';

  const services = [];
  if (status.gemini_available) services.push('Gemini');
  if (status.grok_available) services.push('Grok');
  if (status.memory_enabled) services.push('Memory');

  if (services.length === 0) return 'No services available';
  return services.join(' + ');
};

// Theme and styling helpers
export const getStatusColor = (status) => {
  switch (status.toLowerCase()) {
    case 'error':
      return 'text-red-500 bg-red-50';
    case 'warning':
      return 'text-yellow-600 bg-yellow-50';
    case 'success':
      return 'text-green-500 bg-green-50';
    case 'info':
      return 'text-blue-500 bg-blue-50';
    default:
      return 'text-gray-500 bg-gray-50';
  }
};

// Message processing helpers
export const processMessageContent = (content) => {
  if (!content) return '';

  // Remove unnecessary whitespace
  content = content.trim();

  // Ensure code blocks are properly formatted
  content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    return `\`\`\`${lang || ''}\n${code.trim()}\n\`\`\``;
  });

  return content;
};

// File validation
export const validateFile = (file, options = {}) => {
  const defaultOptions = {
    maxSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: ['application/pdf'],
    allowedExtensions: ['.pdf']
  };

  const opts = { ...defaultOptions, ...options };
  const errors = [];

  if (file.size > opts.maxSize) {
    errors.push(`File size must be less than ${formatFileSize(opts.maxSize)}`);
  }

  if (!opts.allowedTypes.includes(file.type)) {
    errors.push('Invalid file type');
  }

  const extension = file.name.toLowerCase().match(/\.[^.]*$/)?.[0];
  if (!extension || !opts.allowedExtensions.includes(extension)) {
    errors.push('Invalid file extension');
  }

  return {
    valid: errors.length === 0,
    errors
  };
};

