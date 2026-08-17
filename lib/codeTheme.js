export const codeTheme = {
  name: "learn-docs-dark",
  type: "dark",
  fg: "#c9d1d9",
  bg: "#0d0d15",
  colors: {
    "editor.background": "#0d0d15",
    "editor.foreground": "#c9d1d9",
  },
  tokenColors: [
    {
      scope: ["keyword", "keyword.control", "keyword.operator", "storage", "storage.type", "keyword.declaration"],
      settings: { foreground: "#c792ea" },
    },
    {
      scope: ["string", "string.quoted", "string.regexp", "string.template", "string.other"],
      settings: { foreground: "#a5e075" },
    },
    {
      scope: ["entity.name.function", "support.function", "meta.function-call", "entity.name.method"],
      settings: { foreground: "#82aaff" },
    },
    {
      scope: ["constant.numeric", "constant.language", "constant.other", "variable.language", "support.constant", "entity.name.constant"],
      settings: { foreground: "#f78c6c" },
    },
    {
      scope: ["comment", "comment.block", "comment.line"],
      settings: { foreground: "#6a6a8e", fontStyle: "italic" },
    },
    {
      scope: ["entity.name.class", "entity.name.type", "support.class", "entity.name.type.class"],
      settings: { foreground: "#ffcb6b" },
    },
    {
      scope: ["entity.other.attribute-name", "support.type.property-name", "variable.parameter", "variable.object.property"],
      settings: { foreground: "#82aaff" },
    },
    {
      scope: ["punctuation", "punctuation.separator", "meta.brace", "meta.delimiter"],
      settings: { foreground: "#9c9cb0" },
    },
    {
      scope: ["markup.deleted", "invalid", "invalid.illegal"],
      settings: { foreground: "#fb7185" },
    },
    {
      scope: ["markup.inserted"],
      settings: { foreground: "#34d399" },
    },
    {
      scope: ["markup.bold"],
      settings: { fontStyle: "bold" },
    },
    {
      scope: ["markup.italic"],
      settings: { fontStyle: "italic" },
    },
    {
      scope: ["meta.embedded", "source.embedded", "punctuation.section.embedded"],
      settings: { foreground: "#c9d1d9" },
    },
  ],
};