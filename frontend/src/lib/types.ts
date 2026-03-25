// ============ Documents ============

export interface Document {
  id: string;
  name: string;
  pages: number;
  chunks: number;
  status: "processing" | "ready" | "error";
  created_at: string;
}

export interface DocumentListResponse {
  documents: Document[];
}

// ============ Chat ============

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  error?: string;
}

export interface Source {
  index: number;
  document_id: string;
  document_name: string;
  page: number;
  chunk_text: string;
  relevance_score: number;
}

// ============ SSE Events ============

export interface SSEChunkEvent {
  type: "text";
  content: string;
}

export interface SSESourcesEvent {
  sources: Source[];
}

export interface SSEErrorEvent {
  type: "error";
  message: string;
}
