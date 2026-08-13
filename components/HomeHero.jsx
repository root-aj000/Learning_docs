"use client";

import { useCallback } from "react";
import { Search, Sparkles, Command, BookOpen, Library, Infinity as InfinityIcon } from "lucide-react";

export default function HomeHero({ totalDocs, totalCategories }) {
  const openPalette = useCallback(() => {
    window.dispatchEvent(new CustomEvent("open:command"));
  }, []);

  const stats = [
    { icon: BookOpen, value: totalDocs, label: "Documents" },
    { icon: Library, value: totalCategories, label: "Categories" },
    { icon: InfinityIcon, value: "∞", label: "Knowledge" },
  ];

  return (
    <section className="home-hero">
      <div className="home-hero-badge">
        <Sparkles size={13} />
        Knowledge Base
      </div>

      <h1>
        Learn. Note. <span>Master.</span>
      </h1>

      <p className="home-hero-sub">
        Curated notes on Machine Learning, Deep Learning, LLMs and more — beautifully
        rendered with interactive diagrams and full-text search.
      </p>

      <button className="home-search" onClick={openPalette} aria-label="Search documentation">
        <Search size={17} />
        <span>Search documentation…</span>
        <kbd>
          <Command size={12} />K
        </kbd>
      </button>

      <div className="home-stats">
        {stats.map(({ icon: Icon, value, label }) => (
          <div key={label} className="home-stat">
            <div className="home-stat-value">
              <Icon size={15} className="home-stat-icon" />
              {value}
            </div>
            <div className="home-stat-label">{label}</div>
          </div>
        ))}
      </div>
    </section>
  );
}