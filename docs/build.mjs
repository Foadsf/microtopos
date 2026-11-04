// Build the MicroTopos website: AsciiDoc pages (pages/*.adoc) -> static HTML in _site/.
//   cd docs && npm ci && npm run build
// Each page is converted by Asciidoctor.js and placed in one shared template (header, page outline,
// previous/next links, footer), so every page looks and navigates the same.
import Asciidoctor from "@asciidoctor/core";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(here, "_site");
const adoc = Asciidoctor();

const pyproject = fs.readFileSync(path.join(here, "..", "pyproject.toml"), "utf8");
const version = (pyproject.match(/^version\s*=\s*"([^"]+)"/m) || [])[1] || "0.0.0";
const repo = "https://github.com/Foadsf/microtopos";

// reading order: drives the navigation bar, the tutorial hub and the previous/next links
const PAGES = [
  { file: "index", nav: "Home" },
  { file: "install", nav: "Install" },
  { file: "quick-start", nav: "Quick start" },
  { file: "tutorials", nav: "Tutorials" },
  { file: "tutorial-height-map", parent: "tutorials" },
  { file: "tutorial-mosaic", parent: "tutorials" },
  { file: "tutorial-calibration", parent: "tutorials" },
  { file: "manual", nav: "Manual" },
  { file: "metrology", nav: "Metrology" },
  { file: "troubleshooting", nav: "Help" },
  { file: "development", nav: "Develop" },
];

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const strip = (s) => String(s).replace(/<[^>]+>/g, "");

function outline(doc) {
  const items = [];
  const walk = (secs, depth) => {
    for (const s of secs) {
      if (s.getLevel() > 2) continue;
      items.push(`<li class="l${s.getLevel()}"><a href="#${s.getId()}">${s.getTitle()}</a></li>`);
      if (depth < 1) walk(s.getSections(), depth + 1);
    }
  };
  walk(doc.getSections(), 0);
  return items.length > 2 ? `<nav class="outline" aria-label="On this page"><p>On this page</p><ul>${items.join("")}</ul></nav>` : "";
}

function navbar(current) {
  const cur = PAGES.find((p) => p.file === current);
  const active = cur.parent || current;
  const links = PAGES.filter((p) => p.nav && p.file !== "index")
    .map((p) => `<a href="${p.file}.html"${p.file === active ? ' aria-current="page"' : ""}>${p.nav}</a>`)
    .join("");
  return `<header class="site-header"><div class="bar">
  <a class="brand" href="index.html" aria-label="MicroTopos home"><img src="images/logo.svg" alt="" width="28" height="28"><span>MicroTopos</span></a>
  <button class="nav-toggle" aria-expanded="false" aria-controls="site-nav" aria-label="Menu"><span></span><span></span><span></span></button>
  <nav id="site-nav" class="site-nav">${links}<a class="gh" href="${repo}" aria-label="Source on GitHub">GitHub</a>
  <button class="theme-toggle" type="button" aria-label="Switch light or dark theme" title="Light / dark"></button></nav>
</div></header>`;
}

function prevNext(i) {
  const seq = PAGES;
  const prev = i > 0 ? seq[i - 1] : null;
  const next = i < seq.length - 1 ? seq[i + 1] : null;
  const t = (p) => esc(p.title);
  return `<nav class="prev-next">${prev ? `<a class="prev" href="${prev.file}.html"><small>Previous</small>${t(prev)}</a>` : "<span></span>"}${
    next ? `<a class="next" href="${next.file}.html"><small>Next</small>${t(next)}</a>` : "<span></span>"}</nav>`;
}

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

// load every page first so titles are known for the previous/next links
const docs = PAGES.map((p) => {
  const src = path.join(here, "pages", `${p.file}.adoc`);
  const doc = adoc.loadFile(src, {
    safe: "safe",
    attributes: {
      version, repo, "release-url": `${repo}/releases/tag/v${version}`, imagesdir: "images",
      "sectanchors": "", "idprefix": "", "idseparator": "-", "experimental": "", "figure-caption!": "",
      "table-caption!": "", "example-caption!": "", "icons!": "",
    },
  });
  p.title = strip(doc.getDocumentTitle() || p.file);
  p.description = doc.getAttribute("description") || "MicroTopos: microscope stitching, height maps and surface metrology";
  return doc;
});

PAGES.forEach((p, i) => {
  const doc = docs[i];
  const body = doc.convert();
  const isHome = p.file === "index";
  const title = isHome ? `MicroTopos — microscope stitching, height maps and surface metrology` : `${p.title} · MicroTopos`;
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(p.description)}">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(p.description)}">
<meta property="og:image" content="images/overview-dark.png">
<link rel="icon" href="images/logo.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/site.css">
<script>try{const t=localStorage.getItem("mt-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
</head>
<body class="page-${p.file}${isHome ? " home" : ""}">
<a class="skip" href="#main">Skip to content</a>
${navbar(p.file)}
<div class="layout${isHome ? " wide" : ""}">
<main id="main" class="doc">
${isHome ? "" : `<h1 class="page-title">${doc.getDocumentTitle()}</h1>`}
${body}
${isHome ? "" : prevNext(i)}
</main>
${isHome ? "" : outline(doc)}
</div>
<footer class="site-footer"><div class="bar">
<p><strong>MicroTopos ${esc(version)}</strong> · free software under the GNU GPL v3 or later</p>
<p><a href="${repo}">Source</a> · <a href="${repo}/releases">Releases</a> · <a href="${repo}/issues">Report a problem</a> · <a href="development.html">Develop</a></p>
</div></footer>
<div class="lightbox" hidden><img alt=""><p></p></div>
<script src="assets/site.js" defer></script>
</body>
</html>
`;
  fs.writeFileSync(path.join(out, `${p.file}.html`), html);
});

fs.cpSync(path.join(here, "images"), path.join(out, "images"), { recursive: true });
fs.cpSync(path.join(here, "assets"), path.join(out, "assets"), { recursive: true });
fs.writeFileSync(path.join(out, ".nojekyll"), "");
console.log(`built ${PAGES.length} pages for MicroTopos ${version} -> ${path.relative(process.cwd(), out) || out}`);
