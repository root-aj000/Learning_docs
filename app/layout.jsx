import "./globals.css";
import "katex/dist/katex.min.css";
import AppShell from "../components/AppShell";
import { getMarkdownPaths, getDocSummaries } from "../lib/docs";

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata = {
  title: {
    default: "LearnDocs — Knowledge Base",
    template: "%s · LearnDocs",
  },
  description:
    "A premium documentation viewer for ML, DL, LLM, and more. Browse markdown docs with Mermaid diagram support and full-text search.",
};

const themeInit = `(function(){try{var t=localStorage.getItem("theme");if(!t){t=window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";}document.documentElement.dataset.theme=t;}catch(e){}})();`;

export default function RootLayout({ children }) {
  const paths = getMarkdownPaths();
  const summaries = getDocSummaries();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>
        <AppShell docs={paths} summaries={summaries}>
          {children}
        </AppShell>
      </body>
    </html>
  );
}