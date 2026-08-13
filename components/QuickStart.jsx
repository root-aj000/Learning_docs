import Link from "next/link";
import { getDocSummaries } from "../lib/docs";
import { Rocket, FileText, ArrowRight } from "lucide-react";

const QUICK_LINKS = ["ml/numpy", "ml/machine_learning", "ml/pytorch"];

export default function QuickStart() {
  const docs = getDocSummaries();
  const picks = QUICK_LINKS.map((p) => docs.find((d) => d.path === p))
    .filter(Boolean)
    .slice(0, 3);

  if (picks.length === 0) return null;

  return (
    <section className="quickstart">
      <div className="quickstart-head">
        <span className="quickstart-icon">
          <Rocket size={16} />
        </span>
        <div>
          <h2 className="quickstart-title">Jump right in</h2>
          <p className="quickstart-sub">Popular starting points</p>
        </div>
      </div>
      <div className="quickstart-links">
        {picks.map((doc) => (
          <Link
            key={doc.path}
            href={`/docs/${encodeURI(doc.path)}`}
            className="quickstart-link"
          >
            <FileText size={14} />
            <span className="quickstart-link-text">
              <span className="quickstart-link-title">{doc.title}</span>
              <span className="quickstart-link-desc">{doc.description || doc.category}</span>
            </span>
            <ArrowRight size={14} className="quickstart-link-arrow" />
          </Link>
        ))}
      </div>
    </section>
  );
}