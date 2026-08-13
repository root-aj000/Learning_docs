"use client";

import { useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { ChevronDown, FileText, FolderOpen, ArrowRight } from "lucide-react";
import { getIcon } from "./iconMap";

export default function CategoryExplorer({ categories, activeCategory, onSelect }) {
  const sectionRef = useRef(null);
  const currentRef = useRef(null);

  const scrollToActive = useCallback(() => {
    if (currentRef.current && activeCategory) {
      currentRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [activeCategory]);

  useEffect(() => {
    if (activeCategory) scrollToActive();
  }, [activeCategory, scrollToActive]);

  const cat = categories.find((c) => c.key === activeCategory) || null;

  return (
    <section className="explorer" ref={sectionRef}>
      <div className="section-head">
        <div>
          <h2 className="section-title">All documents</h2>
          <p className="section-sub">Every note, organised by topic</p>
        </div>
        <span className="section-total">
          {categories.reduce((s, c) => s + c.totalFiles, 0)} docs
        </span>
      </div>

      {activeCategory && (
        <div className="explorer-crumb">
          <span>Showing</span>
          <button className="explorer-crumb-cat" onClick={() => onSelect(cat?.key)}>
            {cat?.label}
          </button>
          <button className="explorer-crumb-clear" onClick={() => onSelect(null)}>
            Clear filter ×
          </button>
        </div>
      )}

      <div className="explorer-list">
        {categories.map((category) => {
          const open = category.key === activeCategory;
          const Icon = getIcon(category.icon);
          return (
            <div
              key={category.key}
              ref={open ? currentRef : null}
              className={`explorer-item ${open ? "open" : ""}`}
            >
              <button
                className="explorer-item-header"
                onClick={() => onSelect(open ? null : category.key)}
                aria-expanded={open}
              >
                <span className="explorer-item-icon">
                  <Icon size={16} />
                </span>
                <span className="explorer-item-label">{category.label}</span>
                <span className="explorer-item-count">{category.totalFiles}</span>
                <ChevronDown size={15} className="explorer-item-chevron" />
              </button>

              {open && (
                <div className="explorer-item-body">
                  {category.subfolders.map((sf) => (
                    <div key={sf.key} className="explorer-group">
                      <div className="explorer-group-title">
                        <FolderOpen size={13} />
                        {sf.label}
                        <span className="explorer-group-count">{sf.docs.length}</span>
                      </div>
                      <div className="explorer-group-list">
                        {sf.docs.map((doc) => (
                          <Link
                            key={doc.path}
                            href={`/docs/${encodeURI(doc.path)}`}
                            className="explorer-doc"
                          >
                            <FileText size={13} className="explorer-doc-icon" />
                            <span className="explorer-doc-title">{doc.title}</span>
                            {doc.description && (
                              <span className="explorer-doc-desc">{doc.description}</span>
                            )}
                            <ArrowRight size={13} className="explorer-doc-arrow" />
                          </Link>
                        ))}
                      </div>
                    </div>
                  ))}
                  <div className="explorer-group">
                    <div className="explorer-group-list">
                      {category.files.map((doc) => (
                        <Link
                          key={doc.path}
                          href={`/docs/${encodeURI(doc.path)}`}
                          className="explorer-doc"
                        >
                          <FileText size={13} className="explorer-doc-icon" />
                          <span className="explorer-doc-title">{doc.title}</span>
                          {doc.description && (
                            <span className="explorer-doc-desc">{doc.description}</span>
                          )}
                          <ArrowRight size={13} className="explorer-doc-arrow" />
                        </Link>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}