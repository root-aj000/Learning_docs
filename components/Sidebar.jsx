"use client";

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  Brain,
  Bot,
  BarChart3,
  Cpu,
  GraduationCap,
  Map,
  Search,
  X,
  ChevronRight,
  Folder,
  FolderOpen,
  FileText,
  Layers,
  Shield,
  Scale,
  Zap,
  FlaskConical,
  FileCheck,
  Package,
  Command,
  Sparkles,
} from "lucide-react";

const ICONS = {
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

const FOLDER_LABELS = {
  ml: "Machine Learning",
  MLDL: "ML & DL Theory",
  LLM: "Large Language Models",
  clg: "College Notes",
  compt: "Computing",
  DL: "Deep Learning",
  cyber: "Cyber Security",
  daa: "Design & Analysis of Algo",
  hpc: "High Performance Comp.",
  st: "Software Testing",
  nvidia: "NVIDIA",
  answer: "Answers",
  answers: "Answers",
  software: "Software",
};

function prettifyName(name) {
  return FOLDER_LABELS[name] || name
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildTree(docs) {
  const root = { children: {}, files: [] };

  docs.forEach((docPath) => {
    const parts = docPath.split("/");
    let node = root;

    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i];
      if (!node.children[seg]) {
        node.children[seg] = { children: {}, files: [] };
      }
      node = node.children[seg];
    }

    const fileName = parts[parts.length - 1];
    node.files.push({
      path: docPath,
      name: prettifyName(fileName),
    });
  });

  function toArray(node) {
    const childEntries = Object.entries(node.children)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => ({
        key,
        label: prettifyName(key),
        icon: key,
        ...toArray(child),
      }));

    const sortedFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
    return { folders: childEntries, files: sortedFiles };
  }

  return toArray(root);
}

function FolderNode({ folder, pathname, depth, onLinkClick, isOpen, setIsOpen }) {
  const Icon = ICONS[folder.key] || Folder;
  const childrenCount = folder.folders.reduce((s, f) => s + f.files.length + f.folders.length, 0) + folder.files.length;

  return (
    <div className="nav-folder">
      <button
        className={`nav-folder-header ${isOpen ? "expanded" : ""}`}
        onClick={() => setIsOpen(!isOpen)}
        style={{ paddingLeft: 12 + depth * 14 }}
        aria-expanded={isOpen}
      >
        <span className="nav-folder-chevron">
          <ChevronRight size={13} />
        </span>
        <span className="nav-folder-icon">
          <Icon size={15} />
        </span>
        <span className="nav-folder-label">{folder.label}</span>
        <span className="nav-count">{childrenCount}</span>
      </button>

      {isOpen && (
        <div className="nav-folder-children">
          {folder.folders.map((sub) => (
            <SidebarFolder
              key={sub.key}
              folder={sub}
              pathname={pathname}
              depth={depth + 1}
              onLinkClick={onLinkClick}
            />
          ))}
          {folder.files.map((item) => {
            const href = `/docs/${encodeURI(item.path)}`;
            const isActive = pathname === href || decodeURIComponent(pathname) === href;
            return (
              <Link
                key={item.path}
                href={href}
                className={`nav-link ${isActive ? "active" : ""}`}
                style={{ paddingLeft: 22 + depth * 14 }}
                onClick={onLinkClick}
              >
                <FileText size={13} className="nav-link-icon" />
                <span className="nav-link-text">{item.name}</span>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SidebarFolder(props) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const isAncestorActive =
      props.pathname.includes(`/docs/${props.folder.key}/`) ||
      props.folder.files.some(
        (f) => props.pathname === `/docs/${encodeURI(f.path)}` || decodeURIComponent(props.pathname) === `/docs/${f.path}`
      ) ||
      props.folder.folders.some((sub) => props.pathname.includes(`/docs/${props.folder.key}/${sub.key}`));
    if (isAncestorActive) setOpen(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.pathname]);

  return <FolderNode {...props} isOpen={open} setIsOpen={setOpen} />;
}

export default function Sidebar({ docs, isOpen, onClose }) {
  const pathname = usePathname();
  const [searchQuery, setSearchQuery] = useState("");
  const [topLevel, setTopLevel] = useState({});
  const searchRef = useRef(null);

  const tree = useMemo(() => buildTree(docs), [docs]);

  useEffect(() => {
    const initial = {};
    tree.folders.forEach((folder) => {
      initial[folder.key] = true;
    });
    setTopLevel(initial);
  }, [tree]);

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

  const toggleTopLevel = useCallback((key) => {
    setTopLevel((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        // handled by global palette; just close mobile drawer
        if (window.innerWidth < 1024) onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? "visible" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <Link href="/" className="sidebar-logo" onClick={onClose}>
            <div className="sidebar-logo-icon">
              <BookOpen size={17} />
            </div>
            <div className="sidebar-logo-texts">
              <div className="sidebar-logo-text">LearnDocs</div>
              <div className="sidebar-logo-sub">Knowledge Base</div>
            </div>
          </Link>
          <button className="sidebar-close" onClick={onClose} aria-label="Close menu">
            <X size={16} />
          </button>
        </div>

        <div className="sidebar-search">
          <div className="search-input-wrap">
            <Search size={15} className="search-input-icon" />
            <input
              ref={searchRef}
              className="search-input"
              type="text"
              placeholder="Filter docs…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoComplete="off"
              aria-label="Filter documentation"
            />
            {searchQuery && (
              <button className="search-clear" onClick={() => setSearchQuery("")} aria-label="Clear filter">
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        <nav className="sidebar-nav" role="navigation" aria-label="Documentation navigation">
          {filteredTree.folders.length === 0 && filteredTree.files.length === 0 ? (
            <div className="no-results">
              <Search size={20} className="no-results-icon" />
              <p>
                {searchQuery.trim() ? `No results for “${searchQuery}”` : "No docs found"}
              </p>
            </div>
          ) : (
            <>
              {filteredTree.folders.map((folder) => (
                <div key={folder.key} className="nav-folder">
                  <button
                    className={`nav-folder-header top ${topLevel[folder.key] ? "expanded" : ""}`}
                    onClick={() => toggleTopLevel(folder.key)}
                    aria-expanded={topLevel[folder.key]}
                  >
                    <span className="nav-folder-icon">
                      {topLevel[folder.key] ? <FolderOpen size={15} /> : <Folder size={15} />}
                    </span>
                    <span className="nav-folder-label">{folder.label}</span>
                    <span className="nav-count">
                      {folder.files.length + folder.folders.length}
                    </span>
                  </button>

                  {topLevel[folder.key] && (
                    <div className="nav-folder-children">
                      {folder.folders.map((sub) => (
                        <SidebarFolder
                          key={sub.key}
                          folder={sub}
                          pathname={pathname}
                          depth={1}
                          onLinkClick={onClose}
                        />
                      ))}
                      {folder.files.map((item) => {
                        const href = `/docs/${encodeURI(item.path)}`;
                        const isActive = pathname === href || decodeURIComponent(pathname) === href;
                        return (
                          <Link
                            key={item.path}
                            href={href}
                            className={`nav-link ${isActive ? "active" : ""}`}
                            style={{ paddingLeft: 34 }}
                            onClick={onClose}
                          >
                            <FileText size={13} className="nav-link-icon" />
                            <span className="nav-link-text">{item.name}</span>
                          </Link>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}

              {filteredTree.files.map((item) => {
                const href = `/docs/${encodeURI(item.path)}`;
                const isActive = pathname === href || decodeURIComponent(pathname) === href;
                return (
                  <Link
                    key={item.path}
                    href={href}
                    className={`nav-link ${isActive ? "active" : ""}`}
                    style={{ paddingLeft: 22 }}
                    onClick={onClose}
                  >
                    <FileText size={13} className="nav-link-icon" />
                    <span className="nav-link-text">{item.name}</span>
                  </Link>
                );
              })}
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-footer-text">
            <Sparkles size={11} />
            {docs.length} documents
          </span>
          <span className="sidebar-footer-kbd">
            <Command size={11} />K to search
          </span>
        </div>
      </aside>
    </>
  );
}