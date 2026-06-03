"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import mermaid from "mermaid";

let mermaidCounter = 0;

function MermaidBlock({ chart }) {
  const containerRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;

    mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      themeVariables: {
        primaryColor: "#7c5cfc",
        primaryTextColor: "#1a1a2e",
        primaryBorderColor: "#5a3fd6",
        lineColor: "#6b7280",
        secondaryColor: "#e8e0ff",
        tertiaryColor: "#f0f4ff",
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: "14px",
      },
      flowchart: { curve: "basis", padding: 16 },
      sequence: { actorMargin: 50, messageMargin: 40 },
    });

    const id = `mermaid-${++mermaidCounter}-${Date.now()}`;

    mermaid
      .render(id, chart.trim())
      .then(({ svg }) => {
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      })
      .catch((err) => {
        console.error("Mermaid render error:", err);
        setError(err?.message || "Failed to render diagram");
      });
  }, [chart]);

  if (error) {
    return (
      <div className="mermaid-block" style={{ color: "#ef4444", fontSize: 13, justifyContent: "flex-start", flexDirection: "column", gap: 8 }}>
        <strong>⚠ Diagram Error</strong>
        <pre style={{ fontSize: 12, whiteSpace: "pre-wrap", color: "#9ca3af" }}>{error}</pre>
      </div>
    );
  }

  return <div className="mermaid-block" ref={containerRef} />;
}

export default function MarkdownViewer({ content }) {
  return (
    <article className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            if (!inline && match?.[1] === "mermaid") {
              return <MermaidBlock chart={String(children).trim()} />;
            }

            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}
