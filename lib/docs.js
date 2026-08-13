import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { normalizeHeadingText } from "./slug";

const ROOT = process.cwd();
const IGNORED_DIRS = new Set(["node_modules", ".git", ".next", ".kilo"]);
const EXCLUDED_FILES = new Set(["README.md"]);

export const FOLDER_META = {
  ml: { label: "Machine Learning", icon: "brain", desc: "NumPy, Pandas, Scikit-Learn, PyTorch & more", accent: "violet" },
  LLM: { label: "Large Language Models", icon: "bot", desc: "LLM architecture & training", accent: "cyan" },
  MLDL: { label: "ML & DL Theory", icon: "chart", desc: "Core theory & concepts", accent: "emerald" },
  clg: { label: "College Notes", icon: "graduation", desc: "DL, DAA, Cyber, HPC, ST & more", accent: "amber" },
  compt: { label: "Computing", icon: "cpu", desc: "NVIDIA & computing fundamentals", accent: "rose" },
  guide: { label: "Project Guide", icon: "map", desc: "End-to-end build guide", accent: "blue" },
};

export const SUBFOLDER_META = {
  DL: { label: "Deep Learning", icon: "layers" },
  cyber: { label: "Cyber Security", icon: "shield" },
  daa: { label: "Design & Analysis of Algorithms", icon: "scale" },
  hpc: { label: "High Performance Computing", icon: "zap" },
  st: { label: "Software Testing", icon: "flask" },
  nvidia: { label: "NVIDIA", icon: "cpu" },
  answer: { label: "Answers", icon: "file-check" },
  answers: { label: "Answers", icon: "file-check" },
  software: { label: "Software", icon: "package" },
};

export function prettifyName(name) {
  if (SUBFOLDER_META[name]?.label) return SUBFOLDER_META[name].label;
  return name.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function isMarkdownFile(fileName) {
  return /\.(md|mdx)$/i.test(fileName);
}

function walkMarkdown(dir, relative = "") {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries.flatMap((entry) => {
    if (IGNORED_DIRS.has(entry.name) || entry.name.startsWith(".")) {
      return [];
    }

    const fullPath = path.join(dir, entry.name);
    const fileRelative = relative ? `${relative}/${entry.name}` : entry.name;

    if (entry.isDirectory()) {
      return walkMarkdown(fullPath, fileRelative);
    }

    if (entry.isFile() && isMarkdownFile(entry.name)) {
      return [{ relative: fileRelative, fullPath }];
    }

    return [];
  });
}

export function getMarkdownFiles() {
  return walkMarkdown(ROOT).filter(
    (item) =>
      !item.relative.startsWith("app/") &&
      !item.relative.startsWith("lib/") &&
      !item.relative.startsWith("components/") &&
      !EXCLUDED_FILES.has(path.basename(item.relative))
  );
}

export function getMarkdownPaths() {
  return getMarkdownFiles().map((item) => item.relative.replace(/\.(md|mdx)$/i, ""));
}

export function fileToSlug(relative) {
  return relative.replace(/\.(md|mdx)$/i, "");
}

function getFileForSlug(slugSegments) {
  const slugArray = Array.isArray(slugSegments) ? slugSegments : [slugSegments];
  const slug = slugArray.join("/");
  const filePathMd = path.join(ROOT, `${slug}.md`);
  const filePathMdx = path.join(ROOT, `${slug}.mdx`);

  if (fs.existsSync(filePathMd)) return filePathMd;
  if (fs.existsSync(filePathMdx)) return filePathMdx;
  return null;
}

function stripFrontmatter(content) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(content);
  return match ? content.slice(match[0].length) : content;
}

export function extractHeadings(content) {
  const lines = stripFrontmatter(content).split("\n");
  const headings = [];
  let inCode = false;

  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inCode = !inCode;
      continue;
    }
    if (inCode) continue;

    const match = /^(#{2,3})\s+(.+)/.exec(line);
    if (!match) continue;
    const level = match[1].length;
    const rawText = normalizeHeadingText(match[2]);
    if (!rawText) continue;
    headings.push({ level, text: rawText });
  }

  return headings;
}

function wordsIn(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

export function getDocSummaries() {
  return getMarkdownFiles().map(({ relative, fullPath }) => {
    const raw = fs.readFileSync(fullPath, "utf8");
    const { data, content } = matter(raw);
    const body = stripFrontmatter(content);
    const category = relative.includes("/") ? relative.split("/")[0] : null;
    const fallbackTitle = prettifyName(path.basename(relative, path.extname(relative)));

    let description = data.description;
    if (!description) {
      const para = body.split("\n\n").find((p) => p.trim() && !/^(#|```|<|!\[|[-*]\s)/.test(p.trim()));
      description = para ? para.replace(/[#*_`]/g, "").trim().slice(0, 160) : "";
    }

    return {
      path: fileToSlug(relative),
      title: data.title ? String(data.title) : fallbackTitle,
      description,
      category,
      readingTime: Math.max(1, Math.round(wordsIn(body) / 200)),
      headings: extractHeadings(content),
    };
  });
}

export function buildCategories(summaries = getDocSummaries()) {
  const topLevel = {};

  summaries.forEach((doc) => {
    const parts = doc.path.split("/");
    const catKey = parts.length > 1 ? parts[0] : "_root";

    if (!topLevel[catKey]) topLevel[catKey] = { subfolders: {}, files: [] };

    if (parts.length <= 2) {
      topLevel[catKey].files.push(doc);
    } else {
      const subKey = parts[1];
      if (!topLevel[catKey].subfolders[subKey]) {
        topLevel[catKey].subfolders[subKey] = [];
      }
      topLevel[catKey].subfolders[subKey].push(doc);
    }
  });

  return Object.entries(topLevel)
    .filter(([key]) => key !== "_root")
    .map(([key, data]) => {
      const meta = FOLDER_META[key] || {};
      const subfolders = Object.entries(data.subfolders)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([sfKey, sfFiles]) => ({
          key: sfKey,
          label: prettifyName(sfKey),
          icon: SUBFOLDER_META[sfKey]?.icon || "folder",
          docs: sfFiles.sort((a, b) => a.title.localeCompare(b.title)),
        }));

      const files = data.files.sort((a, b) => a.title.localeCompare(b.title));
      const totalFiles = files.length + subfolders.reduce((s, f) => s + f.docs.length, 0);

      return {
        key,
        label: meta.label || prettifyName(key),
        icon: meta.icon || "folder",
        accent: meta.accent || "violet",
        desc: meta.desc || `${totalFiles} document${totalFiles !== 1 ? "s" : ""}`,
        files,
        subfolders,
        totalFiles,
      };
    });
}

export function getFlatDocs() {
  return getDocSummaries();
}

export function getDocByPath(pathname) {
  const slug = pathname.replace(/^\/docs\//, "").replace(/\/$/, "");
  const summ = getDocSummaries().find((d) => d.path === decodeURIComponent(slug));
  return summ || null;
}

export function getDocDetail(slugSegments) {
  const filePath = getFileForSlug(slugSegments);
  if (!filePath) {
    const slug = (Array.isArray(slugSegments) ? slugSegments : [slugSegments]).join("/");
    throw new Error(`Markdown file not found: ${slug}`);
  }

  const raw = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter(raw);
  const relative = path.relative(ROOT, filePath).split(path.sep).join("/");
  const summary = getDocSummaries().find((d) => d.path === fileToSlug(relative));
  const slug = fileToSlug(relative);
  const parts = slug.split("/");
  const category = parts.length > 1 ? parts[0] : null;

  return {
    content,
    slug,
    title: summary?.title || path.basename(slug),
    description: summary?.description || "",
    category,
    readingTime: summary?.readingTime || 1,
    headings: extractHeadings(content),
    keyword: data.tags?.join(", ") || "",
  };
}

export function getAdjacentDocs(slug) {
  const summaries = getDocSummaries().filter((d) => d.category === slug.split("/")[0]);
  const sorted = summaries.sort((a, b) => a.path.localeCompare(b.path));
  const idx = sorted.findIndex((d) => d.path === slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? sorted[idx - 1] : null,
    next: idx < sorted.length - 1 ? sorted[idx + 1] : null,
  };
}