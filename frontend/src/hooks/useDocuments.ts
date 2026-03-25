"use client";

import { useState, useEffect, useCallback } from "react";
import { Document } from "@/lib/types";
import { listDocuments, uploadDocument, deleteDocument } from "@/lib/api";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch documents on mount
  const refresh = useCallback(async () => {
    try {
      setIsLoading(true);
      const docs = await listDocuments();
      setDocuments(docs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Upload
  const upload = useCallback(async (file: File) => {
    try {
      setIsUploading(true);
      setError(null);
      const doc = await uploadDocument(file);
      setDocuments((prev) => [doc, ...prev]);
      return doc;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setError(msg);
      throw err;
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Delete
  const remove = useCallback(async (id: string) => {
    try {
      await deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
      throw err;
    }
  }, []);

  return {
    documents,
    isLoading,
    isUploading,
    error,
    upload,
    remove,
    refresh,
  };
}
