"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Menu, Search, Moon, Sun, ChevronRight, Home } from "lucide-react";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";

function prettifySegment(seg) {
  const labels = {
    docs: "Docs",
    ml: "Machine Learning",
    LLM: "LLMs",
    MLDL: "Theory",
    clg: "College Notes",
    compt: "Computing",
    guide: "Project Guide",
    DL: "Deep Learning",
    cyber: "Cyber Security",
    daa: "Algorithms",
    hpc: "HPC",
    st: "Software Testing",
    nvidia: "NVIDIA",
    answer: "Answers",
    answers: "Answers",
  };
  return labels[seg] || seg.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function ThemeToggle() {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const resolved = stored || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    setTheme(resolved);
    document.documentElement.dataset.theme = resolved;
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("theme", next);
      return next;
    });
  }, []);

  return (
    <button
      className="header-icon-btn theme-toggle"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}

export default function AppShell({ docs, summaries, children }) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (sidebarOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [sidebarOpen]);

  const onScroll = useCallback(() => {
    const scrollTop = window.scrollY;
    const height = document.documentElement.scrollHeight - window.innerHeight;
    setProgress(height > 0 ? Math.min(100, (scrollTop / height) * 100) : 0);
  }, []);

  useEffect(() => {
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [onScroll]);

  const segments = pathname.split("/").filter(Boolean);
  const isDocPage = segments[0] === "docs";

  const crumbs = [
    { label: "Home", href: "/" },
    ...segments.map((seg, i) => {
      const visible = isDocPage ? i > 0 : true;
      return {
        label: prettifySegment(decodeURIComponent(seg)),
        href: "/" + segments.slice(0, i + 1).join("/"),
        isFile: isDocPage && i === segments.length - 1,
        hidden: !visible,
      };
    }),
  ].filter((c) => !c.hidden);

  const openPalette = useCallback(() => {
    window.dispatchEvent(new CustomEvent("open:command"));
  }, []);

  return (
    <div className="app-layout">
      <Sidebar
        docs={docs}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="main-content">
        <div
          className="reading-progress"
          style={{ width: `${progress}%` }}
          aria-hidden="true"
        />

        <header className="content-header">
          <div className="content-header-inner">
            <button
              className="header-icon-btn mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={18} />
            </button>

            <nav className="breadcrumb" aria-label="Breadcrumb">
              {crumbs.map((item, i) => (
                <span key={item.href} className="breadcrumb-item">
                  {i > 0 && <ChevronRight size={13} className="breadcrumb-sep" />}
                  {i < crumbs.length - 1 ? (
                    <Link href={item.href} className="breadcrumb-link">
                      {i === 0 ? <Home size={13} className="breadcrumb-home" /> : item.label}
                    </Link>
                  ) : (
                    <span className="breadcrumb-current">{item.label}</span>
                  )}
                </span>
              ))}
            </nav>

            <div className="header-actions">
              <button className="search-trigger" onClick={openPalette}>
                <Search size={15} />
                <span className="search-trigger-label">Search</span>
                <kbd>⌘K</kbd>
              </button>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="content-body">{children}</div>
      </div>

      <CommandPalette docs={summaries || []} />
    </div>
  );
}