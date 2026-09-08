// Minimal Markdown -> docx converter tailored to docs/desktop_manual_ko.md's
// own subset of Markdown (headings, tables, bullet/numbered lists, bold,
// inline code, fenced code blocks, blockquotes, hr). Not a general parser.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, ShadingType, TabStopType, TabStopPosition,
  ExternalHyperlink, LevelFormat, convertInchesToTwip, ImageRun,
  Footer, PageNumber, TableOfContents,
} = require("docx");

const srcPath = process.argv[2];
const outPath = process.argv[3];
const srcDir = path.dirname(srcPath);
const md = fs.readFileSync(srcPath, "utf8");
const lines = md.split(/\r?\n/);

function getPngDimensions(buffer) {
  // IHDR chunk is fixed at offset 16 (8-byte signature + 4-byte length + "IHDR").
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

const FONT = "Malgun Gothic"; // reliable Hangul-capable font on Windows/Word
const BODY_SIZE = 21; // half-points (10.5pt)

function parseInlineSpecs(text) {
  // Returns plain descriptor objects (not TextRun instances -- docx-js's
  // TextRun doesn't expose its construction options back out, so building
  // one TextRun from another's "options" silently produces an empty run).
  const specs = [];
  const re = /(\*\*(.+?)\*\*|`([^`]+?)`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) specs.push({ text: text.slice(last, m.index) });
    if (m[2] !== undefined) {
      specs.push({ text: m[2], bold: true });
    } else if (m[3] !== undefined) {
      specs.push({ text: m[3], code: true });
    }
    last = re.lastIndex;
  }
  if (last < text.length) specs.push({ text: text.slice(last) });
  if (specs.length === 0) specs.push({ text: "" });
  return specs;
}

function specsToRuns(specs, overrides = {}) {
  return specs.map(s => s.code
    ? new TextRun({ text: s.text, font: "Consolas", size: BODY_SIZE - 1, ...overrides })
    : new TextRun({ text: s.text, bold: !!s.bold || !!overrides.bold, font: FONT, size: BODY_SIZE, ...overrides }));
}

function parseInline(text) {
  return specsToRuns(parseInlineSpecs(text));
}

function stripLinks(text) {
  // [label](#anchor) -> label -- anchor navigation isn't reliable once
  // converted to Word, and every heading is already a real Heading style,
  // so plain text is the safer choice here for our own "목차" list.
  return text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1");
}

const children = [];
const numbering = {
  config: [
    {
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.7), hanging: convertInchesToTwip(0.2) } } } },
      ],
    },
    {
      reference: "numbers",
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.25) } } } },
      ],
    },
  ],
};

function heading(text, level) {
  children.push(new Paragraph({
    heading: level,
    spacing: { before: 280, after: 140 },
    children: specsToRuns(parseInlineSpecs(text), { bold: true }),
  }));
}

function bodyPara(text, opts = {}) {
  children.push(new Paragraph({ spacing: { after: 120 }, children: parseInline(text), ...opts }));
}

function bulletPara(text, level = 0) {
  children.push(new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60 },
    children: parseInline(text),
  }));
}

function numberedPara(text) {
  children.push(new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 60 },
    children: parseInline(text),
  }));
}

function imageWithCaption(altText, relPath) {
  const abs = path.resolve(srcDir, relPath);
  const buf = fs.readFileSync(abs);
  const { width, height } = getPngDimensions(buf);
  const maxW = 5.8, maxH = 4.2; // inches, fits within the page margins set below
  const scale = Math.min(maxW / (width / 96), maxH / (height / 96), 1);
  const dispW = (width / 96) * scale;
  const dispH = (height / 96) * scale;
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({
      data: buf,
      type: "png",
      // docx-js's ImageRun transformation is in pixels (96 DPI equivalent,
      // converted to EMU internally) -- NOT twips.
      transformation: { width: Math.round(dispW * 96), height: Math.round(dispH * 96) },
    })],
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: altText, italics: true, font: FONT, size: BODY_SIZE - 3, color: "555555" })],
  }));
}

function codeBlock(codeLines) {
  children.push(new Paragraph({
    spacing: { before: 80, after: 160 },
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
    },
    children: codeLines.flatMap((l, i) => {
      const run = new TextRun({ text: l || " ", font: "Consolas", size: BODY_SIZE - 1 });
      return i === 0 ? [run] : [new TextRun({ break: 1 }), run];
    }),
  }));
}

function hr() {
  children.push(new Paragraph({
    spacing: { before: 160, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999" } },
    children: [new TextRun({ text: "" })],
  }));
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "E7ECF3" } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: specsToRuns(parseInlineSpecs(text), opts.header ? { bold: true } : {}) })],
  });
}

function table(rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  children.push(new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => cell(c, { header: ri === 0, width: colWidths[ci] })),
    })),
  }));
  children.push(new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }));
}

let i = 0;
let inToc = false; // see the "## 목차" special-case below
while (i < lines.length) {
  const line = lines[i];

  if (line.startsWith("# ")) { inToc = false; heading(line.slice(2), HeadingLevel.TITLE); i++; continue; }
  if (line.startsWith("### ")) { inToc = false; heading(line.slice(4), HeadingLevel.HEADING_3); i++; continue; }
  if (line.startsWith("## ")) {
    inToc = false;
    const text = line.slice(3);
    heading(text, HeadingLevel.HEADING_2);
    i++;
    if (text.trim() === "목차") {
      // Word can compute and paginate this itself (and keep it clickable)
      // far better than a hand-typed list of markdown anchor links, which
      // don't resolve once converted -- skip the .md's own numbered list
      // right below this heading (still useful as a plain-text fallback
      // in the .md/GitHub rendering) and insert a native TOC field
      // instead. `features.updateFields` above asks Word to fill it in on
      // open; F9 forces it in a reader that ignores that.
      inToc = true;
      children.push(new TableOfContents("", { hyperlink: true, headingStyleRange: "2-3" }));
      children.push(new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }));
    }
    continue;
  }

  if (inToc && /^\d+\.\s/.test(line)) { i++; continue; }

  if (line.trim() === "---") { hr(); i++; continue; }

  const imgMatch = /^!\[(.*?)\]\((.+?)\)\s*$/.exec(line);
  if (imgMatch) { imageWithCaption(imgMatch[1], imgMatch[2]); i++; continue; }

  if (line.startsWith("> ")) {
    const buf = [];
    while (i < lines.length && lines[i].startsWith(">")) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
    children.push(new Paragraph({
      indent: { left: convertInchesToTwip(0.3) },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: "8FAADC" } },
      spacing: { after: 160 },
      shading: { type: ShadingType.CLEAR, fill: "F3F6FB" },
      children: specsToRuns(parseInlineSpecs(buf.join(" ")), { italics: true }),
    }));
    continue;
  }

  if (line.startsWith("```")) {
    i++;
    const buf = [];
    while (i < lines.length && !lines[i].startsWith("```")) { buf.push(lines[i]); i++; }
    i++; // skip closing fence
    codeBlock(buf);
    continue;
  }

  if (/^\|.*\|\s*$/.test(line)) {
    const rows = [];
    while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) {
      const raw = lines[i];
      if (!/^\|[\s:|-]+\|\s*$/.test(raw)) {
        const cells = raw.slice(1, raw.length - 1).split("|").map(s => s.trim());
        rows.push(cells);
      }
      i++;
    }
    const nCols = rows[0].length;
    const pageWidth = 9350; // usable width in DXA at ~0.65in margins on A4-ish
    const first = Math.round(pageWidth * (nCols <= 2 ? 0.28 : 0.22));
    const rest = Math.round((pageWidth - first) / (nCols - 1));
    const widths = [first, ...Array(nCols - 1).fill(rest)];
    table(rows, widths);
    continue;
  }

  if (/^\d+\.\s/.test(line)) {
    while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
      numberedPara(lines[i].replace(/^\d+\.\s/, ""));
      i++;
    }
    continue;
  }

  if (/^-\s/.test(line) || /^\s{2,}-\s/.test(line)) {
    while (i < lines.length && (/^-\s/.test(lines[i]) || /^\s{2,}-\s/.test(lines[i]))) {
      const indented = /^\s{2,}-\s/.test(lines[i]);
      bulletPara(lines[i].replace(/^\s*-\s/, ""), indented ? 1 : 0);
      i++;
    }
    continue;
  }

  if (line.trim() === "") { i++; continue; }

  bodyPara(line);
  i++;
}

const doc = new Document({
  numbering,
  // Ask Word to refresh field results when the document is opened. Without
  // it the table-of-contents field renders empty until a reader hits F9, and
  // Word silently populates and re-saves it the first time anyone opens the
  // file -- which shows up as an uncommitted change to a generated artifact.
  features: { updateFields: true },
  styles: {
    default: {
      document: { run: { font: FONT, size: BODY_SIZE } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4 in DXA
        margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 }, // ~0.8in
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: BODY_SIZE - 3 }),
            new TextRun({ text: " / ", font: FONT, size: BODY_SIZE - 3 }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: BODY_SIZE - 3 }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  console.log("Wrote", outPath, buf.length, "bytes");
});
