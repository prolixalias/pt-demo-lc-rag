import React, { useState } from 'react';
import { Bug, ChevronDown, ChevronUp, Clock, Cpu, AlertTriangle } from 'lucide-react';

const DebugSection = ({ title, content, icon: Icon }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border-b border-gray-200 last:border-0">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          {Icon && <Icon className="w-4 h-4 text-gray-500" />}
          <span className="text-sm font-medium text-gray-700">{title}</span>
        </div>
        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {isExpanded && (
        <div className="p-2 text-xs font-mono bg-gray-50">
          <pre className="whitespace-pre-wrap overflow-x-auto">
            {typeof content === 'object'
              ? JSON.stringify(content, null, 2)
              : content}
          </pre>
        </div>
      )}
    </div>
  );
};

export const DebugSettingsPanel = ({ debugInfo }) => {
  if (!debugInfo) return null;

  const sections = [
    {
      title: 'Timing Information',
      icon: Clock,
      content: {
        process_start: debugInfo.process_start,
        completion_time: debugInfo.completion_time,
        total_duration: debugInfo.total_duration
      }
    },
    {
      title: 'Model Performance',
      icon: Cpu,
      content: {
        tokens_used: debugInfo.tokens_used,
        model_version: debugInfo.model_version,
        temperature: debugInfo.temperature
      }
    }
  ];

  // Add error section if there are any errors
  if (debugInfo.error) {
    sections.push({
      title: 'Errors',
      icon: AlertTriangle,
      content: debugInfo.error
    });
  }

  return (
    <div className="mt-4 bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="p-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <Bug className="w-4 h-4 text-yellow-600" />
          <h3 className="text-sm font-medium text-gray-700">Debug Information</h3>
        </div>
      </div>

      <div className="divide-y divide-gray-200">
        {sections.map((section, index) => (
          <DebugSection
            key={index}
            title={section.title}
            content={section.content}
            icon={section.icon}
          />
        ))}
      </div>
    </div>
  );
};

export default DebugSettingsPanel;