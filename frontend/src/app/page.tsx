"use client";

import { useDocuments } from "@/hooks/useDocuments";
import { useChat } from "@/hooks/useChat";
import { UploadPanel } from "@/components/UploadPanel";
import { DocumentList } from "@/components/DocumentList";
import { MessageList } from "@/components/MessageList";
import { ChatInput } from "@/components/ChatInput";

export default function Home() {
  const docs = useDocuments();
  const chat = useChat();

  const readyDocs = docs.documents.filter((d) => d.status === "ready");
  const hasDocuments = readyDocs.length > 0;

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <aside className="w-80 border-r bg-white flex flex-col">
        <div className="p-4 border-b">
          <h1 className="text-lg font-semibold text-gray-900">
            Chat with Docs
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Upload PDFs and ask questions
          </p>
        </div>

        <div className="p-4 flex-1 overflow-y-auto space-y-4">
          <UploadPanel
            onUpload={docs.upload}
            isUploading={docs.isUploading}
          />

          {docs.error && (
            <p className="text-red-500 text-xs">{docs.error}</p>
          )}

          <DocumentList
            documents={docs.documents}
            onDelete={docs.remove}
          />
        </div>
      </aside>

      {/* Chat area */}
      <main className="flex-1 flex flex-col bg-white">
        <MessageList messages={chat.messages} />

        <ChatInput
          onSend={(question) => {
            const docIds = readyDocs.map((d) => d.id);
            chat.sendMessage(question, docIds);
          }}
          isStreaming={chat.isStreaming}
          onStop={chat.stop}
          disabled={!hasDocuments}
        />

        {chat.error && (
          <p className="text-red-500 text-xs text-center pb-2">
            {chat.error}
          </p>
        )}
      </main>
    </div>
  );
}
