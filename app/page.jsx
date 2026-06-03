import Link from "next/link";
import { getMarkdownFiles } from "../lib/docs";

const FOLDER_META = {
  ml: { label: "Machine Learning", icon: "🧠", desc: "NumPy, Pandas, Scikit-Learn, PyTorch & more" },
  MLDL: { label: "ML & DL Theory", icon: "📊", desc: "Core theory & concepts" },
  LLM: { label: "Large Language Models", icon: "🤖", desc: "LLM architecture & training" },
  clg: { label: "College Notes", icon: "🎓", desc: "DL, DAA, Cyber, HPC, ST & more" },
  compt: { label: "Computing", icon: "💻", desc: "NVIDIA & computing fundamentals" },
};

const SUBFOLDER_LABELS = {
  DL: "Deep Learning",
  cyber: "Cyber Security",
  daa: "Design & Analysis of Algorithms",
  hpc: "High Performance Computing",
  st: "Software Testing",
  nvidia: "NVIDIA",
  answer: "Answers",
  answers: "Answers",
  software: "Software",
};

function prettifyName(name) {
  return SUBFOLDER_LABELS[name] || name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Build nested tree from flat file list, returning top-level categories
 * with their sub-folder structure preserved.
 */
function buildCategories(files) {
  const topLevel = {};

  files.forEach(({ relative }) => {
    const stripped = relative.replace(/\.(md|mdx)$/i, "");
    const parts = stripped.split("/");
    const catKey = parts.length > 1 ? parts[0] : "_root";

    if (!topLevel[catKey]) topLevel[catKey] = { subfolders: {}, files: [] };

    if (parts.length <= 2) {
      // Direct file under category (or root)
      topLevel[catKey].files.push({
        path: stripped,
        name: prettifyName(parts[parts.length - 1]),
      });
    } else {
      // Nested: parts[1] is subfolder, rest is path within
      const subKey = parts[1];
      if (!topLevel[catKey].subfolders[subKey]) {
        topLevel[catKey].subfolders[subKey] = [];
      }
      topLevel[catKey].subfolders[subKey].push({
        path: stripped,
        name: parts.slice(2).map(prettifyName).join(" / "),
        shortName: prettifyName(parts[parts.length - 1]),
      });
    }
  });

  return Object.entries(topLevel).map(([key, data]) => {
    const subfolderEntries = Object.entries(data.subfolders)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([sfKey, sfFiles]) => ({
        key: sfKey,
        label: prettifyName(sfKey),
        files: sfFiles.sort((a, b) => a.shortName.localeCompare(b.shortName)),
      }));

    const totalFiles = data.files.length + Object.values(data.subfolders).reduce((s, f) => s + f.length, 0);

    return {
      key,
      label: FOLDER_META[key]?.label || prettifyName(key),
      icon: FOLDER_META[key]?.icon || "📁",
      desc: FOLDER_META[key]?.desc || `${totalFiles} document${totalFiles !== 1 ? "s" : ""}`,
      files: data.files.sort((a, b) => a.name.localeCompare(b.name)),
      subfolders: subfolderEntries,
      totalFiles,
    };
  });
}

export default function Home() {
  const files = getMarkdownFiles();
  const categories = buildCategories(files);
  const totalDocs = files.length;
  const totalCategories = categories.filter((c) => c.key !== "_root").length;

  return (
    <>
      <div className="home-hero">
        <div className="home-hero-badge">✨ Knowledge Base</div>
        <h1>Your Learning<br />Documentation Hub</h1>
        <p>
          Browse through curated notes on Machine Learning, Deep Learning,
          LLMs, and more — all beautifully rendered with Mermaid diagram support.
        </p>
        <div className="home-stats">
          <div className="home-stat">
            <div className="home-stat-value">{totalDocs}</div>
            <div className="home-stat-label">Documents</div>
          </div>
          <div className="home-stat">
            <div className="home-stat-value">{totalCategories}</div>
            <div className="home-stat-label">Categories</div>
          </div>
          <div className="home-stat">
            <div className="home-stat-value">∞</div>
            <div className="home-stat-label">Knowledge</div>
          </div>
        </div>
      </div>

      <div className="home-grid">
        {categories.map((cat) => (
          <div key={cat.key} className="category-card animate-in">
            <div className="category-card-icon">{cat.icon}</div>
            <div className="category-card-title">{cat.label}</div>
            <div className="category-card-count">{cat.desc}</div>

            <div className="category-card-items">
              {/* Show subfolders as groups */}
              {cat.subfolders.map((sf) => (
                <div key={sf.key} className="category-subfolder">
                  <div className="category-subfolder-header">
                    📂 {sf.label}
                    <span className="category-subfolder-count">{sf.files.length}</span>
                  </div>
                  {sf.files.slice(0, 3).map((item) => (
                    <Link
                      key={item.path}
                      href={`/docs/${encodeURI(item.path)}`}
                      className="category-card-item sub-item"
                    >
                      {item.shortName}
                    </Link>
                  ))}
                  {sf.files.length > 3 && (
                    <div className="category-card-item sub-item" style={{ color: "var(--accent)", fontWeight: 500 }}>
                      + {sf.files.length - 3} more
                    </div>
                  )}
                </div>
              ))}

              {/* Direct files */}
              {cat.files.slice(0, 5).map((item) => (
                <Link
                  key={item.path}
                  href={`/docs/${encodeURI(item.path)}`}
                  className="category-card-item"
                >
                  {item.name}
                </Link>
              ))}
              {cat.files.length > 5 && (
                <div className="category-card-item" style={{ color: "var(--accent)", fontWeight: 500 }}>
                  + {cat.files.length - 5} more
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
