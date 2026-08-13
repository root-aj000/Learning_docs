"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Search,
  FileText,
  Command,
  ArrowRight,
  GraduationCap,
  Brain,
  Cpu,
  Bot,
  BarChart3,
  Map,
  Folder,
  Layers,
  Shield,
  Scale,
  Zap,
  FlaskConical,
  FileCheck,
  Package,
  ScrollText,
  Sparkles,
} from "lucide-react";

const CATEGORY_ICONS = {
  ml: Brain,
  LLM: Bot,
  MLDL: BarChart3,
  clg: GraduationCap,
  compt: Cpu,
  guide: Map,
  DL: Layers,
  cyber: Shield,
  daa: Scale,
  hpc: Zap,
  st: FlaskConical,
  nvidia: Cpu,
  answer: FileCheck,
  answers: FileCheck,
  software: Package,
};

function categoryLabel(key) {
  const labels = {
    ml: "Machine Learning",
    LLM: "LLMs",
    MLDL: "Theory",
    clg: "College",
    compt: "Computing",
    guide: "Guide",
    DL: "Deep Learning",
    cyber: "Security",
    daa: "Algorithms",
    hpc: "HPC",
    st: "Testing",
    nvidia: "NVIDIA",
    answer: "Answers",
    answers: "Answers",
    software: "Software",
  };
  return labels[key] || key;
}

function Highlight({ text, query }) {
  if (!query) return text;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark>{text.slice(idx, idx + query.length)}</mark>
      {text.slice(idx + query.length)}
    </>
  );
}

export default function CommandPalette({ docs }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const openPalette = useCallback(() => {
    setQuery("");
    setActiveIndex(0);
    setOpen(true);
    setTimeout(() => inputRef.current?.focus(), 30);
  }, []);

  const closePalette = useCallback(() => {
    setOpen(false);
    setQuery("");
    setActiveIndex(0);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        closePalette();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, closePalette]);

  useEffect(() => {
    const onGlobalKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        openPalette();
      }
    };
    const onOpenEvent = () => openPalette();
    window.addEventListener("keydown", onGlobalKey);
    window.addEventListener("open:command", onOpenEvent);
    return () => {
      window.removeEventListener("keydown", onGlobalKey);
      window.removeEventListener("open:command", onOpenEvent);
    };
  }, [openPalette]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      return docs
        .filter((d) => d.category === "ml" || d.category === "guide")
        .slice(0, 6)
        .map((d) => ({ ...d, matchType: "popular" }));
    }
    return docs
      .filter(
        (d) =>
          d.title.toLowerCase().includes(q) ||
          d.path.toLowerCase().includes(q) ||
          (d.category || "").toLowerCase().includes(q) ||
          (d.description || "").toLowerCase().includes(q)
      )
      .slice(0, 10)
      .map((d) => ({ ...d, matchType: "match" }));
  }, [docs, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  useEffect(() => {
    if (!open) return;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  useEffect(() => {
    if (!listRef.current) return;
    const el = listRef.current.querySelector(`[data-index="${activeIndex}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  const onKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      const hit = results[activeIndex];
      if (hit) {
        router.push(`/docs/${encodeURI(hit.path)}`);
        closePalette();
      }
    }
  };

  if (!open) return null;

  return (
    <div className="command-palette" role="dialog" aria-modal="true" aria-label="Search documentation">
      <div className="command-backdrop" onClick={closePalette} />
      <div className="command-dialog">
        <div className="command-input-row">
          <Search size={18} className="command-search-icon" />
          <input
            ref={inputRef}
            className="command-input"
            placeholder="Search documentation…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            autoComplete="off"
            spellCheck={false}
            aria-label="Search documentation"
          />
          <kbd className="command-esc">esc</kbd>
        </div>

        <div className="command-results" ref={listRef}>
          {results.length === 0 ? (
            <div className="command-empty">
              <Search size={22} />
              <p>No documents match “{query}”</p>
              <span>Try a different keyword or category</span>
            </div>
          ) : (
            results.map((doc, i) => {
              const Icon = CATEGORY_ICONS[doc.category] || FileText;
              return (
                <button
                  key={doc.path}
                  data-index={i}
                  className={`command-result ${i === activeIndex ? "active" : ""}`}
                  onMouseEnter={() => setActiveIndex(i)}
                  onClick={() => {
                    router.push(`/docs/${encodeURI(doc.path)}`);
                    closePalette();
                  }}
                >
                  <span className="command-result-icon">
                    <Icon size={15} />
                  </span>
                  <span className="command-result-body">
                    <span className="command-result-title">
                      <Highlight text={doc.title} query={query} />
                    </span>
                    {doc.description && (
                      <span className="command-result-desc">
                        <Highlight text={doc.description} query={query} />
                      </span>
                    )}
                  </span>
                  <span className="command-result-cat">{categoryLabel(doc.category)}</span>
                  {i === activeIndex && <ArrowRight size={14} className="command-result-arrow" />}
                </button>
              );
            })
          )}
        </div>

        <div className="command-footer">
          <span>
            <Sparkles size={12} />
            {query.trim() ? `${results.length} results` : "Popular docs"}
          </span>
          <span className="command-footer-keys">
            <kbd>
              <Command size={11} />K
            </kbd>
            to open
          </span>
        </div>
      </div>
    </div>
  );
}