"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import mermaid from "mermaid";
import { AlertTriangle, Braces } from "lucide-react";
import CopyButton from "./CopyButton";

let mermaidCounter = 0;

export default function MermaidBlock({ chart }) {
  const containerRef = useRef(null);
  const [error, setError] = useState(null);

  const renderDiagram = useCallback((chartStr) => {
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      themeVariables: {
        primaryColor: "#f1ecff",
        primaryTextColor: "#1e1b2e",
        primaryBorderColor: "#7c5cfc",
        lineColor: "#8b8ba3",
        secondaryColor: "#e8fdf8",
        tertiaryColor: "#eef2ff",
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: "14px",
      },
      flowchart: { curve: "basis", padding: 16, htmlLabels: true },
      sequence: { actorMargin: 50, messageMargin: 40 },
    });

    const id = `mermaid-${++mermaidCounter}-${Date.now()}`;

    mermaid
      .render(id, chartStr.trim())
      .then(({ svg }) => {
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      })
      .catch((err) => {
        console.error("Mermaid render error:", err);
        setError(err?.message || "Failed to render diagram");
      });
  }, []);

  useEffect(() => {
    if (!chart) return;
    setError(null);
    renderDiagram(chart);
  }, [chart, renderDiagram]);

  if (error) {
    return (
      <div className="mermaid-block mermaid-error">
        <div className="mermaid-error-head">
          <AlertTriangle size={14} />
          <strong>Diagram error</strong>
        </div>
        <pre className="mermaid-error-body">{error}</pre>
      </div>
    );
  }

  return (
    <div className="mermaid-block">
      <div className="mermaid-header">
        <span className="mermaid-title">
          <Braces size={12} /> Diagram
        </span>
        <CopyButton code={chart} label="diagram" />
      </div>
      <div className="mermaid-canvas" ref={containerRef} />
    </div>
  );
}
