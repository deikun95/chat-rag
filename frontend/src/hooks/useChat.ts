"use client";

import { useState, useCallback, useRef } from "react";
import { ChatMessage, Source } from "@/lib/types";
import { streamChat } from "@/lib/api";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (question: string, documentIds?: string[]) => {
      // Build history from existing messages (exclude streaming/error ones)
      const history = messages
        .filter((m) => !m.isStreaming && !m.error && m.content)
        .map((m) => ({ role: m.role, content: m.content }));

      // Add user message
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: question,
      };

      // Add empty assistant message (will be filled by stream)
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setError(null);
      setIsStreaming(true);

      // Abort controller for cancellation
      abortRef.current = new AbortController();

      try {
        const response = await streamChat(
          question,
          documentIds,
          history,
          abortRef.current.signal,
        );

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        if (!response.body) {
          throw new Error("No response body");
        }

        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events from buffer
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          let currentEvent = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              const dataStr = line.slice(6);
              try {
                const data = JSON.parse(dataStr);

                if (currentEvent === "chunk" && data.content) {
                  // Append text token
                  setMessages((prev) => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last?.isStreaming) {
                      updated[updated.length - 1] = {
                        ...last,
                        content: last.content + data.content,
                      };
                    }
                    return updated;
                  });
                } else if (currentEvent === "sources") {
                  // Attach sources to the message
                  setMessages((prev) => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last) {
                      updated[updated.length - 1] = {
                        ...last,
                        sources: data.sources as Source[],
                      };
                    }
                    return updated;
                  });
                } else if (currentEvent === "done") {
                  // Mark streaming complete
                  setMessages((prev) => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last) {
                      updated[updated.length - 1] = {
                        ...last,
                        isStreaming: false,
                      };
                    }
                    return updated;
                  });
                } else if (currentEvent === "error") {
                  setMessages((prev) => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last) {
                      updated[updated.length - 1] = {
                        ...last,
                        isStreaming: false,
                        error: data.message,
                      };
                    }
                    return updated;
                  });
                }
              } catch {
                // Ignore malformed JSON
              }
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;

        const msg = err instanceof Error ? err.message : "Connection failed";
        setError(msg);
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.isStreaming) {
            updated[updated.length - 1] = {
              ...last,
              isStreaming: false,
              error: msg,
            };
          }
          return updated;
        });
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [messages],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, sendMessage, isStreaming, error, stop, clear };
}
