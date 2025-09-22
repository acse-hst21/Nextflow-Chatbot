// components/Chat.tsx
"use client";

import React, { useEffect, useState } from "react";
import MessageBubble from "./MessageBubble";

const BACKEND_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";;

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

export default function Chat() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [appStatus, setAppStatus] = useState<{mock_mode: boolean; message: string} | null>(null);

  // Create session and get app status on mount
  useEffect(() => {
    (async () => {
      console.log("Frontend attempting to connect to backend at:", BACKEND_BASE);
      try {
        // Create session
        const sessionRes = await fetch(`${BACKEND_BASE}/api/session`, { method: "POST" });
        const sessionData = await sessionRes.json();
        setSessionId(sessionData.session_id);

        // Get app status
        const statusRes = await fetch(`${BACKEND_BASE}/api/status`);
        const statusData = await statusRes.json();
        console.log("Received status from backend:", statusData);
        setAppStatus(statusData);
      } catch (e) {
        console.error("Failed to initialize app. Backend URL:", BACKEND_BASE, "Error:", e);
        // If backend is unreachable, show error message
        setAppStatus({
          mock_mode: true,
          message: "Backend unreachable - please try again later"
        });
      }
    })();
  }, []);

  async function sendMessage(e?: React.FormEvent) {
    e?.preventDefault();
    if (!input.trim() || !sessionId) return;

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: input }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: input }),
      });

      if (!response.body) {
        console.error("No response body from backend");
        setIsLoading(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantReply = "";

      // Add placeholder assistant message
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Split by lines and process each SSE event
        const lines = chunk.split('\n');
        let currentEvent = 'message'; // default event type

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7); // Remove 'event: ' prefix
          } else if (line.startsWith('data: ')) {
            const text = line.slice(6); // Remove 'data: ' prefix
            if (text === '[DONE]') continue;

            // Handle different event types
            if (currentEvent === 'metadata') {
              // Parse and attach metadata to the last assistant message
              try {
                const metadata = JSON.parse(text);
                setMessages((prev) => {
                  const updated = [...prev];
                  if (updated.length > 0 && updated[updated.length - 1].role === "assistant") {
                    updated[updated.length - 1] = {
                      ...updated[updated.length - 1],
                      citations: metadata.citations || []
                    };
                  }
                  return updated;
                });
              } catch (e) {
                console.error("Failed to parse metadata:", e);
              }
            } else {
              // Regular message content
              assistantReply += text;

              // Update last assistant message
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: "assistant", content: assistantReply };
                return updated;
              });
            }
            // Reset event type after processing
            currentEvent = 'message';
          }
        }
      }
    } catch (err) {
      console.error("Streaming error", err);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div>
      {/* App Status Notification */}
      {appStatus && (
        <div
          style={{
            marginBottom: 12,
            padding: "8px 12px",
            borderRadius: 8,
            backgroundColor: appStatus.mock_mode ? "#fff3cd" : "#d1edff",
            border: `1px solid ${appStatus.mock_mode ? "#ffeaa7" : "#74b9ff"}`,
            fontSize: 14
          }}
        >
          <strong>{appStatus.mock_mode ? "⚠️ Mock Mode" : "✅ Live Mode"}:</strong> {appStatus.message}
        </div>
      )}

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 13, color: "#666" }}>
          Session: {sessionId ?? <em>creating...</em>}
        </div>
      </div>

      <div
        style={{
          border: "1px solid #eee",
          borderRadius: 10,
          padding: 12,
          height: "60vh",
          overflow: "auto",
        }}
      >
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
      </div>

      <form onSubmit={sendMessage} style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={2}
          placeholder="Ask a question..."
          style={{ flex: 1, padding: 12, borderRadius: 8, border: "1px solid #ddd" }}
        />
        <div style={{ display: "flex", flexDirection: "column" }}>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            style={{ padding: "10px 16px", marginBottom: 8 }}
          >
            {isLoading ? "Thinking..." : "Send"}
          </button>
          {isLoading && (
            <button
              type="button"
              onClick={() => window.location.reload()}
              style={{ padding: "8px 12px" }}
            >
              Stop
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
