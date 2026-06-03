"use client";

import { useState, useMemo, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const FOLDER_ICONS = {
  ml: "🧠",
  MLDL: "📊",
  LLM: "🤖",
  clg: "🎓",
  compt: "💻",
  DL: "🔬",
  cyber: "🛡️",
  daa: "📐",
  hpc: "⚡",
  st: "🧪",
  nvidia: "🎮",
  answer: "📝",
  answers: "📝",
  software: "💿",
};

const FOLDER_LABELS = {
  ml: "Machine Learning",
  MLDL: "ML & DL Theory",
  LLM: "Large Language Models",
  clg: "College Notes",
  compt: "Computing",
  DL: "Deep Learning",
  cyber: "Cyber Security",
  daa: "Design & Analysis of Algo",
  hpc: "High Perf. Computing",
  st: "Software Testing",
  nvidia: "NVIDIA",
  answer: "Answers",
  answers: "Answers",
  software: "Software",
};

function prettifyName(name) {
  return FOLDER_LABELS[name] || name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Build a nested tree from flat doc paths.
 * Each node: { name, label, icon, children: [...nodes], files: [...{path, name}] }
 */
function buildTree(docs) {
  const root = { children: {}, files: [] };

  docs.forEach((docPath) => {
    const parts = docPath.split("/");
    let node = root;

    // Walk through folder segments
    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i];
      if (!node.children[seg]) {
        node.children[seg] = { children: {}, files: [] };
      }
      node = node.children[seg];
    }

    // Last segment is the file name
    const fileName = parts[parts.length - 1];
    node.files.push({
      path: docPath,
      name: prettifyName(fileName),
    });
  });

  // Convert children objects to sorted arrays recursively
  function toArray(node) {
    const childEntries = Object.entries(node.children)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => ({
        key,
        label: prettifyName(key),
        icon: FOLDER_ICONS[key] || "📁",
        ...toArray(child),
      }));

    const sortedFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));

    return { folders: childEntries, files: sortedFiles };
  }

  return toArray(root);
}

function FolderNode({ folder, pathname, depth, onLinkClick, expandedMap, toggleExpand }) {
  const isExpanded = expandedMap[folder.key] !== false; // default open
  const totalItems = folder.files.length + folder.folders.length;

  return (
    <div className="nav-folder" style={{ "--depth": depth }}>
      <button
        className={`nav-folder-header ${isExpanded ? "expanded" : ""}`}
        onClick={() => toggleExpand(folder.key)}
        style={{ paddingLeft: 12 + depth * 16 }}
      >
        <span className="nav-folder-icon">{folder.icon}</span>
        <span className="nav-folder-label">{folder.label}</span>
        <span className="nav-folder-count">{totalItems}</span>
        <svg className={`chevron ${isExpanded ? "open" : ""}`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
        </svg>
      </button>

      {isExpanded && (
        <div className="nav-folder-children">
          {/* Sub-folders first */}
          {folder.folders.map((sub) => (
            <FolderNode
              key={sub.key}
              folder={sub}
              pathname={pathname}
              depth={depth + 1}
              onLinkClick={onLinkClick}
              expandedMap={expandedMap}
              toggleExpand={toggleExpand}
            />
          ))}
          {/* Then files */}
          {folder.files.map((item) => {
            const href = `/docs/${encodeURI(item.path)}`;
            const isActive = pathname === href || decodeURIComponent(pathname) === href;
            return (
              <Link
                key={item.path}
                href={href}
                className={`nav-link ${isActive ? "active" : ""}`}
                style={{ paddingLeft: 16 + (depth + 1) * 16 }}
                onClick={onLinkClick}
              >
                <span className="nav-link-dot" />
                {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function Sidebar({ docs, isOpen, onClose }) {
  const pathname = usePathname();
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedMap, setExpandedMap] = useState({});

  const tree = useMemo(() => buildTree(docs), [docs]);

  const toggleExpand = useCallback((key) => {
    setExpandedMap((prev) => ({ ...prev, [key]: prev[key] === false ? true : false }));
  }, []);

  // Filter: when searching, flatten to matching files grouped by top-level folder
  const filteredTree = useMemo(() => {
    if (!searchQuery.trim()) return tree;
    const q = searchQuery.toLowerCase();

    function filterNode(node) {
      const matchingFiles = node.files.filter(
        (f) => f.name.toLowerCase().includes(q) || f.path.toLowerCase().includes(q)
      );
      const matchingFolders = node.folders
        .map((folder) => {
          const filtered = filterNode(folder);
          if (filtered) return { ...folder, ...filtered };
          // Also match on folder label
          if (folder.label.toLowerCase().includes(q)) return folder;
          return null;
        })
        .filter(Boolean);

      if (matchingFiles.length === 0 && matchingFolders.length === 0) return null;
      return { folders: matchingFolders, files: matchingFiles };
    }

    const result = filterNode(tree);
    return result || { folders: [], files: [] };
  }, [tree, searchQuery]);

  return (
    <>
      <div className={`sidebar-overlay ${isOpen ? "open" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <Link href="/" className="sidebar-logo" onClick={onClose}>
            <div className="sidebar-logo-icon">L</div>
            <div>
              <div className="sidebar-logo-text">LearnDocs</div>
              <div className="sidebar-logo-sub">Knowledge Base</div>
            </div>
          </Link>
        </div>

        <div className="sidebar-search">
          <div className="search-input-wrap">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
            </svg>
            <input
              className="search-input"
              type="text"
              placeholder="Search docs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <nav className="sidebar-nav">
          {filteredTree.folders.length === 0 && filteredTree.files.length === 0 ? (
            <div className="no-results">
              <div className="no-results-icon">🔍</div>
              <div className="no-results-text">No docs found</div>
            </div>
          ) : (
            <>
              {filteredTree.folders.map((folder) => (
                <FolderNode
                  key={folder.key}
                  folder={folder}
                  pathname={pathname}
                  depth={0}
                  onLinkClick={onClose}
                  expandedMap={expandedMap}
                  toggleExpand={toggleExpand}
                />
              ))}
              {/* Root-level files */}
              {filteredTree.files.map((item) => {
                const href = `/docs/${encodeURI(item.path)}`;
                const isActive = pathname === href || decodeURIComponent(pathname) === href;
                return (
                  <Link
                    key={item.path}
                    href={href}
                    className={`nav-link ${isActive ? "active" : ""}`}
                    style={{ paddingLeft: 28 }}
                    onClick={onClose}
                  >
                    <span className="nav-link-dot" />
                    {item.name}
                  </Link>
                );
              })}
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-footer-text">
            {docs.length} documents available
          </div>
        </div>
      </aside>
    </>
  );
}
