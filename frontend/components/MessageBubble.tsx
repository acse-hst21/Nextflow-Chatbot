// components/MessageBubble.tsx
"use client";

import React from "react";

interface Citation {
  id: string;
  title: string;
  url: string;
}

interface Message {
  role: string;
  content: string;
  citations?: Citation[];
}

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        margin: "4px 0",
      }}
    >
      <span
        style={{
          display: "inline-block",
          backgroundColor: isUser ? "#daf1da" : "#f1f1f1",
          color: "#000",
          padding: "8px 12px",
          borderRadius: 12,
          maxWidth: "80%",
          whiteSpace: "pre-wrap", // preserves spaces and line breaks
          wordBreak: "break-word", // prevent overflow
        }}
      >
        {message.content}
      </span>

      {/* Display citations for assistant messages */}
      {!isUser && message.citations && message.citations.length > 0 && (
        <div style={{ marginTop: "4px", maxWidth: "80%" }}>
          <div style={{ fontSize: "12px", color: "#666", marginBottom: "2px" }}>
            Sources:
          </div>
          {message.citations.map((citation, index) => (
            <a
              key={index}
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "inline-block",
                fontSize: "11px",
                color: "#0066cc",
                textDecoration: "none",
                backgroundColor: "#f8f9fa",
                border: "1px solid #e0e0e0",
                borderRadius: "4px",
                padding: "2px 6px",
                marginRight: "4px",
                marginBottom: "2px",
              }}
            >
              {citation.title}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
