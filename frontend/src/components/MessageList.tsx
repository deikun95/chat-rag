"use client";

import { useEffect, useRef } from "react";
import { ChatMessage } from "@/lib/types";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <p className="text-lg">Chat with your documents</p>
          <p className="text-sm mt-1">
            Upload a PDF and ask questions about it
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm ${
              msg.role === "user"
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-900"
            }`}
          >
            {/* Message content */}
            <p className="whitespace-pre-wrap">{msg.content}</p>

            {/* Streaming cursor */}
            {msg.isStreaming && (
              <span className="inline-block w-1.5 h-4 bg-gray-400 animate-pulse ml-0.5" />
            )}

            {/* Error */}
            {msg.error && (
              <p className="text-red-500 text-xs mt-2">Error: {msg.error}</p>
            )}

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-3 pt-2 border-t border-gray-200">
                <p className="text-xs text-gray-500 mb-1">Sources:</p>
                {msg.sources.map((s) => (
                  <div
                    key={s.index}
                    className="text-xs text-gray-500 mb-1 pl-2 border-l-2 border-gray-300"
                  >
                    <span className="font-medium">[{s.index}]</span>{" "}
                    {s.document_name}, p.{s.page}
                    <span className="text-gray-400">
                      {" "}- {(s.relevance_score * 100).toFixed(0)}% match
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
