import { MarkdownAsync } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import { Link as LinkIcon, FileCode2, Terminal } from "lucide-react";
import { slugify, normalizeHeadingText } from "../lib/slug";
import { codeTheme } from "../lib/codeTheme";
import MermaidBlock from "./MermaidBlock";
import CopyButton from "./CopyButton";

function normalizeMath(content) {
  return content.replace(/^[ \t]*\$\$([^\n$][^\n]*?)\$\$[ \t]*$/gm, (_, inner) => `$$\n${inner.trim()}\n$$`);
}

function extractText(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node?.props?.children != null) return extractText(node.props.children);
  return "";
}

const prettyCodeOptions = {
  theme: codeTheme,
  keepBackground: false,
  bypassInlineCode: true,
  defaultLang: "text",
  grid: false,
};

function withLineNumbers(original) {
  let n = 0;
  const lines = [];
  original.forEach((child, i) => {
    if (typeof child === "string") return;
    n += 1;
    lines.push(
      <span className="line" key={i}>
        <span className="line-number" aria-hidden="true">{n}</span>
        <span className="line-content">{child?.props?.children ?? child}</span>
      </span>
    );
  });
  return lines;
}

function CodeBlock({ language, children }) {
  const codeText = extractText(children);
  const code = Array.isArray(children) ? children[0] : children;
  const showNumbers = code?.props?.["data-line-numbers"] != null;
  const original = code?.props?.children;
  const body =
    showNumbers && Array.isArray(original)
      ? <code {...code.props}>{withLineNumbers(original)}</code>
      : children;

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-block-lang">
          {language === "bash" || language === "sh" || language === "shell" ? <Terminal size={12} /> : <FileCode2 size={12} />}
          {language}
        </span>
        <CopyButton code={codeText} label={language} />
      </div>
      <div className="code-block-body"><pre>{body}</pre></div>
    </div>
  );
}

function headingText(children) {
  return Array.isArray(children)
    ? children
        .map((c) => {
          if (typeof c === "string" || typeof c === "number") return String(c);
          if (c?.props?.children) return headingText(c.props.children);
          return "";
        })
        .join("")
    : String(children || "");
}

export default async function MarkdownViewer({ content, headings = [] }) {
  const idByText = new Map();
  headings.forEach((h) => {
    const key = slugify(normalizeHeadingText(h.text));
    if (!idByText.has(key)) idByText.set(key, h.id);
  });

  const headingRenderer = (level) => {
    return function Heading({ children, ...props }) {
      const text = normalizeHeadingText(headingText(children));
      const id = level === 2 || level === 3 ? idByText.get(slugify(text)) || "" : "";
      return (
        <HeadingTag level={level} id={id} {...props}>
          {id && (
            <a className="heading-anchor" href={`#${id}`} aria-label={`Link to ${text}`} tabIndex={-1}>
              <LinkIcon size={15} />
            </a>
          )}
          {children}
        </HeadingTag>
      );
    };
  };

  const element = await MarkdownAsync({
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [
      [rehypePrettyCode, prettyCodeOptions],
      rehypeRaw,
      [rehypeKatex, { throwOnError: false, errorColor: "#f87171" }],
    ],
    components: {
      h1: headingRenderer(1),
      h2: headingRenderer(2),
      h3: headingRenderer(3),
      h4: headingRenderer(4),
      h5: headingRenderer(5),
      h6: headingRenderer(6),

      pre({ children }) {
        const child = Array.isArray(children) ? children[0] : children;
        const dataLang = child?.props?.["data-language"];
        const className = child?.props?.className || "";
        const match = /language-(\w+)/.exec(className);
        const lang = dataLang || match?.[1];

        if (lang === "mermaid") {
          const text = child?.props?.children;
          const chart =
            typeof text === "string"
              ? text
              : Array.isArray(text)
                ? text.map((c) => (typeof c === "string" ? c : String(c))).join("")
                : String(text || "");
          return <MermaidBlock chart={chart.trim()} />;
        }

        return <CodeBlock language={lang || "plain"}>{children}</CodeBlock>;
      },

      table({ children }) {
        return (
          <div className="table-wrap">
            <table>{children}</table>
          </div>
        );
      },

      img({ src, alt, ...props }) {
        return <img src={src} alt={alt || ""} loading="lazy" className="markdown-img" {...props} />;
      },

      a({ href, children, ...props }) {
        const external = /^https?:\/\//.test(href || "");
        return (
          <a
            href={href}
            {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
            className="markdown-link"
            {...props}
          >
            {children}
          </a>
        );
      },
    },
    children: normalizeMath(content),
  });

  return <article className="markdown-body">{element}</article>;
}

function HeadingTag({ level, id, children, ...props }) {
  const Tag = `h${level}`;
  return (
    <Tag id={id} {...props}>
      {children}
    </Tag>
  );
}
