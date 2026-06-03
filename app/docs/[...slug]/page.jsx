import Link from "next/link";
import MarkdownViewer from "../../../components/MarkdownViewer";
import { getMarkdownPaths, getMarkdownBySlug } from "../../../lib/docs";

export function generateStaticParams() {
  return getMarkdownPaths().map((slug) => ({ slug: slug.split("/") }));
}

export default function DocPage({ params }) {
  const slugArray = Array.isArray(params.slug) ? params.slug : [params.slug];
  const { content, title } = getMarkdownBySlug(slugArray);

  const displayTitle = title
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  const category = slugArray.length > 1 ? slugArray[0] : null;

  return (
    <>
      <Link href="/" className="back-link">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10 12L6 8L10 4" />
        </svg>
        All Documents
      </Link>

      <h1 className="doc-title">{displayTitle}</h1>

      <div className="doc-meta">
        {category && (
          <div className="doc-meta-item">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path d="M1.5 2A1.5 1.5 0 0 0 0 3.5v2h3.879a1.5 1.5 0 0 1 1.06.44l1.122 1.12A1.5 1.5 0 0 0 7.12 7.5H16v-4A1.5 1.5 0 0 0 14.5 2h-5.25a.75.75 0 0 1-.544-.235L7.42 0.44A1.5 1.5 0 0 0 6.36 0H1.5z" />
              <path d="M16 9H7.12a3 3 0 0 1-2.12-.879l-1.122-1.121H0v7.5A1.5 1.5 0 0 0 1.5 16h13a1.5 1.5 0 0 0 1.5-1.5V9z" />
            </svg>
            {category}
          </div>
        )}
        <div className="doc-meta-item">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 4a3 3 0 0 1 3-3h4.5a.5.5 0 0 1 .354.146l3 3A.5.5 0 0 1 15 4.5V13a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V4z" />
          </svg>
          Markdown
        </div>
      </div>

      <MarkdownViewer content={content} />
    </>
  );
}
