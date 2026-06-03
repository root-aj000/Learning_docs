import "./globals.css";
import AppShell from "../components/AppShell";
import { getMarkdownPaths } from "../lib/docs";

export const metadata = {
  title: "LearnDocs — Knowledge Base",
  description: "A premium documentation viewer for ML, DL, LLM, and more. Browse markdown docs with Mermaid diagram support.",
};

export default function RootLayout({ children }) {
  const docs = getMarkdownPaths();

  return (
    <html lang="en">
      <body>
        <AppShell docs={docs}>
          {children}
        </AppShell>
      </body>
    </html>
  );
}
