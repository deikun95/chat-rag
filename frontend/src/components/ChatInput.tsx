"use client";

import { useState, useRef, useEffect } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  isStreaming: boolean;
  onStop: () => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, isStreaming, onStop, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 150) + "px";
    }
  }, [input]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput("");
  };

  return (
    <div className="flex items-end gap-2 p-4 border-t bg-white">
      <textarea
        ref={textareaRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
          }
        }}
        placeholder={
          disabled
            ? "Upload a document to start chatting..."
            : "Ask a question about your documents..."
        }
        disabled={disabled || isStreaming}
        rows={1}
        className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-2.5
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
          disabled:bg-gray-100 disabled:text-gray-400
          text-sm"
      />
      {isStreaming ? (
        <button
          onClick={onStop}
          className="px-4 py-2.5 bg-red-500 text-white rounded-lg text-sm
            hover:bg-red-600 transition-colors"
        >
          Stop
        </button>
      ) : (
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || disabled}
          className="px-4 py-2.5 bg-blue-600 text-white rounded-lg text-sm
            hover:bg-blue-700 transition-colors
            disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Send
        </button>
      )}
    </div>
  );
}
