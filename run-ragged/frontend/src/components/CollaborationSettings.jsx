import React, { useState } from 'react';
import { X, Bug, Trash2, AlertCircle ,Thermometer, MessageSquare } from 'lucide-react';
import { toast } from 'react-toastify';

export const CollaborationSettings = ({ initialSettings, onSave, onClose }) => { // <-- Named export
  const [localSettings, setLocalSettings] = useState(initialSettings);

  const handleReset = async () => {
    if (!localSettings.debug_mode) {
      return; // Safety check
    }
  
    if (confirm('WARNING: This will delete all embeddings data. This action cannot be undone. Are you sure?')) {
      try {
        const response = await fetch('/debug/reset-embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ debug_mode: true })
        });
  
        if (!response.ok) {
          throw new Error(`Reset failed: ${response.statusText}`);
        }
  
        const result = await response.json();
        onToastCallback?.('Database reset completed successfully');
      } catch (error) {
        onToastCallback?.(`Reset failed: ${error.message}`, { type: 'error' });
      }
    }
  };

  // Helper function for temperature inputs
  const TemperatureInput = ({ label, value, onChange, description }) => (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label className="text-sm text-[#4C230A] font-medium">{label}</label>
        <span className="text-sm text-[#4C230A]">{value}</span>
      </div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      {description && (
        <p className="text-xs text-gray-500">{description}</p>
      )}
    </div>
  );

  const handleSave = () => {
    onSave(localSettings);
    toast.success('Settings updated successfully');
    setTimeout(() => {
      onClose();
    }, 100);

  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-[#4C230A]">AI Collaboration Settings</h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-6">
          {/* Developer Experience Section */}
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg border-l-4 border-yellow-400">
            <div className="flex items-center gap-2">
              <Bug className="w-5 h-5 text-yellow-600" />
              <h4 className="font-medium text-[#4C230A]">Developer Experience</h4>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm text-[#4C230A] font-medium">Enable Debug Mode</label>
                <p className="text-xs text-gray-500">Show detailed debugging information in chat</p>
              </div>
              <input
                type="checkbox"
                checked={localSettings.debug_mode}
                onChange={(e) => setLocalSettings(prev => ({
                  ...prev,
                  debug_mode: e.target.checked
                }))}
                className="form-checkbox h-4 w-4 text-yellow-600 rounded border-[#CFB87C]"
              />
            </div>
          </div>

          {/* Model Settings Section */}
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-[#4C230A] mb-4">Model Settings</h4>

            {/* Grok Settings */}
            <div className="space-y-4 border-b border-gray-200 pb-4">
              <div className="flex items-center justify-between">
                <label className="text-sm text-[#4C230A]">Enable Grok</label>
                <input
                  type="checkbox"
                  checked={localSettings.enable_grok}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    enable_grok: e.target.checked
                  }))}
                  className="form-checkbox h-4 w-4 text-[#006838] rounded border-[#CFB87C]"
                />
              </div>

              {localSettings.enable_grok && (
                <TemperatureInput
                  label="Grok Temperature"
                  value={localSettings.grok_temperature}
                  onChange={(value) => setLocalSettings(prev => ({
                    ...prev,
                    grok_temperature: value
                  }))}
                  description="Higher values make output more creative, lower values more focused"
                />
              )}
            </div>

            {/* Gemini Settings */}
            <div className="space-y-4 pt-4">
              <TemperatureInput
                label="Gemini Temperature"
                value={localSettings.gemini_temperature}
                onChange={(value) => setLocalSettings(prev => ({
                  ...prev,
                  gemini_temperature: value
                }))}
                description="Controls response randomness"
              />
            </div>
          </div>

          {/* Conversation Settings */}
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare className="w-5 h-5 text-[#006838]" />
              <h4 className="font-medium text-[#4C230A]">Conversation Settings</h4>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm text-[#4C230A] block mb-1">Max Conversation Turns</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={localSettings.max_conversation_turns}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    max_conversation_turns: parseInt(e.target.value)
                  }))}
                  className="w-full px-3 py-2 text-sm border border-[#CFB87C] rounded-lg"
                />
                <p className="text-xs text-gray-500 mt-1">Number of previous messages to consider for context</p>
              </div>

              <TemperatureInput
                label="Synthesis Temperature"
                value={localSettings.synthesis_temperature}
                onChange={(value) => setLocalSettings(prev => ({
                  ...prev,
                  synthesis_temperature: value
                }))}
                description="Controls creativity in response synthesis"
              />
            </div>
          </div>
        </div>

        {localSettings.debug_mode && (
          <div className="mt-6 pt-6 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-red-600 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  Debug Actions
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Danger zone: These actions can affect system data
                </p>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Reset Embeddings
              </button>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-[#4C230A] border border-[#CFB87C] rounded-lg hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm bg-[#006838] text-white rounded-lg hover:bg-[#004D2C]"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

// export default CollaborationSettings;
