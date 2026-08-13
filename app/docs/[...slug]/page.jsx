import Link from "next/link";
import MarkdownViewer from "../../../components/MarkdownViewer";
import TOC from "../../../components/TOC";
import { getMarkdownPaths, getDocDetail, getAdjacentDocs } from "../../../lib/docs";
import { slugify, slugifyWithIndex } from "../../../lib/slug";
import { ArrowLeft, ArrowRight, Clock, FileText, Folder, ChevronLeft } from "lucide-react";

export function generateStaticParams() {
  return getMarkdownPaths().map((slug) => ({ slug: slug.split("/") }));
}

export default function DocPage({ params }) {
  const slugArray = Array.isArray(params.slug) ? params.slug : [params.slug];
  const doc = getDocDetail(slugArray);
  const { prev, next } = getAdjacentDocs(doc.slug);

  const usedIds = new Set();
  const seenSlugs = new Set();
  const tocHeadings = [];
  for (const h of doc.headings) {
    const base = slugify(h.text);
    if (seenSlugs.has(base)) continue;
    seenSlugs.add(base);
    tocHeadings.push({ ...h, id: slugifyWithIndex(h.text, usedIds) });
  }

  const categoryLabel = doc.category
    ? doc.category.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
    : null;

  return (
    <div className="doc-page">
      <div className="doc-main">
        <Link href="/" className="back-link">
          <ChevronLeft size={14} />
          All documents
        </Link>

        <header className="doc-head">
          <h1 className="doc-title">{doc.title}</h1>
          {doc.description && <p className="doc-description">{doc.description}</p>}

          <div className="doc-meta">
            {categoryLabel && (
              <span className="doc-meta-item">
                <Folder size={13} />
                {categoryLabel}
              </span>
            )}
            <span className="doc-meta-item">
              <Clock size={13} />
              {doc.readingTime} min read
            </span>
            <span className="doc-meta-item">
              <FileText size={13} />
              Markdown
            </span>
          </div>
        </header>

        <MarkdownViewer content={doc.content} headings={tocHeadings} />

        <nav className="doc-pager" aria-label="Document navigation">
          {prev ? (
            <Link href={`/docs/${encodeURI(prev.path)}`} className="doc-pager-card prev">
              <span className="doc-pager-label">
                <ArrowLeft size={13} /> Previous
              </span>
              <span className="doc-pager-title">{prev.title}</span>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={`/docs/${encodeURI(next.path)}`} className="doc-pager-card next">
              <span className="doc-pager-label">
                Next <ArrowRight size={13} />
              </span>
              <span className="doc-pager-title">{next.title}</span>
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </div>

      <aside className="doc-side">
        <TOC headings={tocHeadings} />
      </aside>
    </div>
  );
}