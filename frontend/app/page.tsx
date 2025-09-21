// app/page.tsx
"use client";

import React from "react";
import Chat from "../components/Chat";

export default function Page() {
  return (
    <main style={{ padding: 16, maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ textAlign: "center" }}>Nextflow Chatbot</h1>
      <Chat />
    </main>
  );
}
