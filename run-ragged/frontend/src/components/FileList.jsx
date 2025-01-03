import React from 'react';
import { FileText, Calendar, XCircle } from 'lucide-react';
import { formatFileSize, formatDate } from '../utils';

export const FileList = React.memo(function FileList({ // <-- Named export: export const FileList = ...
  files,
  onDelete,
  loading = false,
  error = null
}) {
  console.log("FileList Rendered", { files, loading, error }); // <-- ADDED: Debug log

  if (loading) {
    return (
      <div className="space-y-2">
        {[...Array(3)].map((_, index) => (
          <div
            key={index}
            className="bg-white rounded-lg shadow animate-pulse"
          >
            <div className="p-3">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-gray-200 rounded-lg" />
                <div className="flex-1">
                  <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
                  <div className="h-3 bg-gray-200 rounded w-1/2" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600">
        {error}
      </div>
    );
  }

  if (!files || files.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex flex-col items-center justify-center py-8">
          <FileText className="w-8 h-8 text-[#006838] mb-4" />
          <p className="text-[#4C230A] text-base">No PDF files found</p>
          <p className="text-[#4C230A]/70 text-sm mt-2">
            Upload a PDF to get started
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {files.map((file) => (
        <div
          key={file.name}
          className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow border border-[#CFB87C]"
        >
          <div className="relative p-3">
            {onDelete && (
              <button
                onClick={() => onDelete(file.name)}
                className="absolute top-1 left-1 text-red-500 hover:text-red-700 transition-colors p-1"
                title="Delete file"
              >
                <XCircle className="w-4 h-4" />
              </button>
            )}

            <div className="flex items-start gap-3 mt-2">
              <div className="p-2 bg-[#E8F5E9] rounded-lg">
                <FileText className="w-5 h-5 text-[#006838]" />
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="text-base font-semibold text-[#4C230A] truncate">
                  {file.name.replace(/\.pdf$/i, '')}
                </h3>

                <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[#4C230A]/70">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {formatDate(file.created, { includeTime: true })}
                  </span>
                  <span>
                    {formatFileSize(file.size)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
});