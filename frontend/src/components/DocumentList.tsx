"use client";

import { Document } from "@/lib/types";

interface DocumentListProps {
  documents: Document[];
  onDelete: (id: string) => void;
}

export function DocumentList({ documents, onDelete }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <p className="text-gray-400 text-sm text-center py-4">
        No documents uploaded yet
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
        >
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-gray-900 truncate">
              {doc.name}
            </p>
            <p className="text-xs text-gray-500">
              {doc.pages} pages · {doc.chunks} chunks ·{" "}
              <span
                className={
                  doc.status === "ready"
                    ? "text-green-600"
                    : doc.status === "error"
                    ? "text-red-600"
                    : "text-yellow-600"
                }
              >
                {doc.status}
              </span>
            </p>
          </div>
          <button
            onClick={() => onDelete(doc.id)}
            className="ml-2 text-gray-400 hover:text-red-500 transition-colors"
            title="Delete document"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
