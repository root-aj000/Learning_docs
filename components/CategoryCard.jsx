"use client";

import { ArrowUpRight, FileText } from "lucide-react";
import { getIcon } from "./iconMap";

const ACCENT = {
  violet: "var(--accent-violet)",
  cyan: "var(--accent-cyan)",
  emerald: "var(--accent-emerald)",
  amber: "var(--accent-amber)",
  rose: "var(--accent-rose)",
  blue: "var(--accent-blue)",
};

export default function CategoryCard({ category, active, onSelect }) {
  const Icon = getIcon(category.icon);
  const accent = ACCENT[category.accent] || ACCENT.violet;

  return (
    <button
      className={`category-card ${active ? "active" : ""}`}
      style={{ "--card-accent": accent }}
      onClick={() => onSelect(category.key)}
      aria-pressed={active}
    >
      <div className="category-card-top">
        <div className="category-card-icon">
          <Icon size={20} />
        </div>
        <span className="category-card-arrow">
          <ArrowUpRight size={15} />
        </span>
      </div>
      <div className="category-card-title">{category.label}</div>
      <div className="category-card-desc">{category.desc}</div>
      <div className="category-card-footer">
        <span className="category-card-count">
          <FileText size={12} />
          {category.totalFiles} doc{category.totalFiles !== 1 ? "s" : ""}
        </span>
        {category.subfolders.length > 0 && (
          <span className="category-card-subs">{category.subfolders.length} subtopics</span>
        )}
      </div>
    </button>
  );
}