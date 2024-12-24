import React, { useState, useRef } from 'react';
import { Upload, RefreshCw, XCircle, AlertCircle } from 'lucide-react';

export const PDFUploadCard = ({ onUploadComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type === 'application/pdf') {
        if (file.size > 200 * 1024 * 1024) { // 200MB limit
          setError('File size must be less than 200MB');
          setSelectedFile(null);
          e.target.value = null;
          return;
        }
        setSelectedFile(file);
        setError(null);
      } else {
        setError('Please select a PDF file');
        setSelectedFile(null);
        e.target.value = null;
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    setUploading(true);
    setError(null);
    setProgress(0);

    try {
      const xhr = new XMLHttpRequest();

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setProgress(Math.round(percentComplete));
        }
      };

      xhr.onload = async () => {
        try {
          if (xhr.status === 200) {
            setSelectedFile(null);
            if (fileInputRef.current) {
              fileInputRef.current.value = '';
            }
            if (typeof onUploadComplete === 'function') {
              await onUploadComplete();
            }
          } else {
            throw new Error(xhr.responseText || 'Upload failed');
          }
        } catch (err) {
          setError(err.message || 'Failed to upload file. Please try again.');
        } finally {
          setUploading(false);
          setProgress(0);
        }
      };

      xhr.onerror = () => {
        setError('Network error occurred');
        setUploading(false);
        setProgress(0);
      };

      xhr.open('POST', '/upload');
      xhr.send(formData);
    } catch (err) {
      setError(err.message || 'Failed to upload file. Please try again.');
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow border border-[#CFB87C] mb-4">
      <div className="p-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#E8F5E9] rounded-lg">
            {uploading ? (
              <RefreshCw className="w-5 h-5 text-[#006838] animate-spin" />
            ) : (
              <Upload className="w-5 h-5 text-[#006838]" />
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-col space-y-2">
              <div className="flex items-center gap-2">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="hidden"
                  ref={fileInputRef}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`
                    px-3 py-1.5 rounded-lg text-sm transition-all duration-200
                    ${uploading
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-[#E8F5E9] text-[#006838] hover:bg-[#006838] hover:text-white'
                    }
                  `}
                  disabled={uploading}
                >
                  Select PDF...
                </button>
                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  className={`
                    flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm
                    transition-all duration-200
                    ${!selectedFile || uploading
                      ? 'bg-[#006838]/50 cursor-not-allowed text-white/90'
                      : 'bg-[#006838] text-white hover:bg-[#004D2C]'
                    }
                  `}
                >
                  {uploading ? (
                    <>
                      <span>{progress}%</span>
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4" />
                      <span>Upload</span>
                    </>
                  )}
                </button>
              </div>

              {selectedFile && (
                <div className="flex items-center gap-2 text-sm text-[#4C230A]/70 animate-[scaleIn_0.2s_ease-out]">
                  <span className="truncate">{selectedFile.name}</span>
                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                    className="text-red-500 hover:text-red-700 transition-colors"
                  >
                    <XCircle size={16} />
                  </button>
                </div>
              )}

              {error && (
                <div className="flex items-center gap-2 text-sm text-red-500">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              {uploading && (
                <div className="w-full h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#006838] transition-all duration-300 ease-out"
                    style={{
                      width: `${progress}%`,
                      transition: 'width 0.3s ease-out'
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PDFUploadCard;
