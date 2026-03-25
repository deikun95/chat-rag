"use client";

import { useCallback, useRef, useState } from "react";

interface UploadPanelProps {
  onUpload: (file: File) => Promise<any>;
  isUploading: boolean;
}

export function UploadPanel({ onUpload, isUploading }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        alert("Only PDF files are supported");
        return;
      }
      await onUpload(file);
    },
    [onUpload],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`
        border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
        transition-colors duration-200
        ${isDragging
          ? "border-blue-500 bg-blue-50"
          : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
        }
        ${isUploading ? "opacity-50 pointer-events-none" : ""}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = ""; // Reset so same file can be re-uploaded
        }}
      />
      {isUploading ? (
        <p className="text-gray-500">Processing document...</p>
      ) : (
        <div>
          <p className="text-gray-600 font-medium">
            Drop a PDF here or click to upload
          </p>
          <p className="text-gray-400 text-sm mt-1">Max 20MB</p>
        </div>
      )}
    </div>
  );
}
