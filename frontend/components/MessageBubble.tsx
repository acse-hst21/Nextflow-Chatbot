// components/MessageBubble.tsx
"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { Components } from "react-markdown";

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

  const components: Components = {
    // Custom link styling
    a: ({ href, children, ...props }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          color: "#0066cc",
          textDecoration: "underline",
        }}
        {...props}
      >
        {children}
      </a>
    ),
    // Custom code block styling
    code: (props) => {
      const { inline, className, children } = props as {
        inline?: boolean;
        className?: string;
        children?: React.ReactNode;
      };
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        <SyntaxHighlighter
          style={tomorrow}
          language={match[1]}
          PreTag="div"
          customStyle={{
            margin: "8px 0",
            borderRadius: "6px",
            fontSize: "13px",
          }}
        >
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      ) : (
        <code
          className={className}
          style={{
            backgroundColor: "#f6f8fa",
            padding: "2px 4px",
            borderRadius: "3px",
            fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
            fontSize: "0.9em",
          }}
        >
          {children}
        </code>
      );
    },
    // Custom paragraph styling
    p: ({ children }) => (
      <p style={{ margin: "8px 0", lineHeight: "1.5" }}>
        {children}
      </p>
    ),
    // Custom list styling
    ul: ({ children }) => (
      <ul style={{ margin: "8px 0", paddingLeft: "20px" }}>
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol style={{ margin: "8px 0", paddingLeft: "20px" }}>
        {children}
      </ol>
    ),
    li: ({ children }) => (
      <li style={{ margin: "2px 0" }}>
        {children}
      </li>
    ),
    // Custom blockquote styling
    blockquote: ({ children }) => (
      <blockquote
        style={{
          borderLeft: "3px solid #d0d7de",
          margin: "8px 0",
          paddingLeft: "16px",
          color: "#656d76",
        }}
      >
        {children}
      </blockquote>
    ),
    // Custom table styling
    table: ({ children }) => (
      <table
        style={{
          borderCollapse: "collapse",
          margin: "8px 0",
          width: "100%",
        }}
      >
        {children}
      </table>
    ),
    th: ({ children }) => (
      <th
        style={{
          border: "1px solid #d0d7de",
          padding: "6px 10px",
          backgroundColor: "#f6f8fa",
          fontWeight: "600",
        }}
      >
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td
        style={{
          border: "1px solid #d0d7de",
          padding: "6px 10px",
        }}
      >
        {children}
      </td>
    ),
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        margin: "4px 0",
      }}
    >
      <div
        style={{
          backgroundColor: isUser ? "#daf1da" : "#f1f1f1",
          color: "#000",
          padding: "8px 12px",
          borderRadius: 12,
          maxWidth: "80%",
          wordBreak: "break-word",
        }}
      >
        {isUser ? (
          // For user messages, keep simple text rendering
          <span style={{ whiteSpace: "pre-wrap" }}>
            {message.content}
          </span>
        ) : (
          // For assistant messages, use Markdown rendering
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={components}
          >
            {message.content}
          </ReactMarkdown>
        )}
      </div>

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
