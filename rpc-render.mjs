import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import rehypePrettyCode from "rehype-pretty-code";

function serialize(node) {
  if (node.type === "text") return node.value.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const props = Object.entries(node.properties || {})
    .filter(([k, v]) => v !== null && v !== "" && k !== "style")
    .map(([k, v]) => ` data-${k}="${v}"`).join("");
  const style = node.properties?.style ? ` style="${node.properties.style}"` : "";
  return `<${node.tagName}${style}${props}>${(node.children || []).map(serialize).join("")}</${node.tagName}>`;
}

const opts = { theme: { light: "github-light", dark: "github-dark" }, keepBackground: false, defaultLang: "text", showLineNumbers: true };
const src = "```js showLineNumbers\nconst x = 1;\nconst y = 2;\n\n// comment\nfunction hi() { return x; }\n```";
const tree = await unified().use(remarkParse).use(remarkRehype)
  .use(rehypePrettyCode, opts)
  .run(unified().use(remarkParse).parse(src));
const pre = tree.children[0].children[0];
console.log(serialize(pre));
