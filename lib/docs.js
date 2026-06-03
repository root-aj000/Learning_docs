import fs from "fs";
import path from "path";

const ROOT = process.cwd();
const IGNORED_DIRS = new Set(["node_modules", ".git", ".next", ".kilo"]);

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
  return walkMarkdown(ROOT).filter((item) => !item.relative.startsWith("app/") && !item.relative.startsWith("lib/") && !item.relative.startsWith("components/"));
}

export function getMarkdownPaths() {
  return getMarkdownFiles().map((item) => item.relative.replace(/\.(md|mdx)$/i, ""));
}

export function getMarkdownBySlug(slugSegments) {
  const slugArray = Array.isArray(slugSegments) ? slugSegments : [slugSegments];
  const slug = slugArray.join("/");
  const filePathMd = path.join(ROOT, `${slug}.md`);
  const filePathMdx = path.join(ROOT, `${slug}.mdx`);

  if (fs.existsSync(filePathMd)) {
    return { content: fs.readFileSync(filePathMd, "utf8"), title: path.basename(slug) };
  }

  if (fs.existsSync(filePathMdx)) {
    return { content: fs.readFileSync(filePathMdx, "utf8"), title: path.basename(slug) };
  }

  throw new Error(`Markdown file not found: ${slug}`);
}
