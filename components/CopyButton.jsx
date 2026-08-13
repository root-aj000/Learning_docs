"use client";

import { useState, useCallback } from "react";
import { Check, Copy } from "lucide-react";

export default function CopyButton({ code, label }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    navigator.clipboard.writeText(code.trim()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  }, [code]);

  return (
    <button className="code-copy" onClick={copy} aria-label={`Copy ${label} code`}>
      {copied ? <Check size={12} /> : <Copy size={12} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}
