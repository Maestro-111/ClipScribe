// Minimal, dependency-free Markdown renderer for chat answers.
//
// The advisory agent replies in Markdown — **bold**, bullet lists, headings,
// `inline code`. The chat used to print `m.content` in a whitespace-pre-wrap
// div, so those markers showed up literally (the stray "*" the user noticed).
// This parses the common subset into React elements — never HTML strings, so
// there is no XSS surface — and degrades gracefully on the partial Markdown
// produced mid-stream (an unclosed `**` just renders as text until it closes).
import { type ReactNode } from "react";

// ── Inline spans: bold, italic, inline code, links ──
// One regex finds the earliest inline marker; we recurse into bold/italic so
// nesting (e.g. bold containing `code`) still renders.
const INLINE =
  /(\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|\*([^*\n]+?)\*|_([^_\n]+?)_|`([^`]+?)`|\[([^\]]+?)\]\((https?:\/\/[^\s)]+)\))/;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let i = 0;
  while (rest.length) {
    const m = INLINE.exec(rest);
    if (!m) {
      out.push(rest);
      break;
    }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    const key = `${keyPrefix}-${i++}`;
    const bold = m[2] ?? m[3];
    const italic = m[4] ?? m[5];
    if (bold !== undefined) {
      out.push(<strong key={key}>{renderInline(bold, key)}</strong>);
    } else if (italic !== undefined) {
      out.push(<em key={key}>{renderInline(italic, key)}</em>);
    } else if (m[6] !== undefined) {
      out.push(
        <code
          key={key}
          className="rounded bg-neutral-100 px-1 py-0.5 font-mono text-[0.85em] text-neutral-800"
        >
          {m[6]}
        </code>,
      );
    } else if (m[7] !== undefined) {
      out.push(
        <a
          key={key}
          href={m[8]}
          target="_blank"
          rel="noreferrer"
          className="text-blue-600 underline hover:text-blue-700"
        >
          {m[7]}
        </a>,
      );
    }
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

// Soft line breaks inside a paragraph/list item: a single newline becomes <br>.
function withBreaks(text: string, keyPrefix: string): ReactNode[] {
  const lines = text.split("\n");
  return lines.flatMap((line, i) => {
    const content = renderInline(line, `${keyPrefix}-l${i}`);
    return i < lines.length - 1
      ? [...content, <br key={`${keyPrefix}-br${i}`} />]
      : content;
  });
}

// ── Block parser: headings, lists, code fences, blockquotes, paragraphs ──
type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "code"; text: string }
  | { kind: "quote"; text: string }
  | { kind: "p"; text: string };

const UL = /^\s*[-*+]\s+(.*)$/;
const OL = /^\s*\d+[.)]\s+(.*)$/;
const HEADING = /^(#{1,6})\s+(.*)$/;

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      blocks.push({ kind: "p", text: para.join("\n") });
      para = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    // Fenced code block: gather until the closing fence (or end-of-stream).
    if (/^\s*```/.test(line)) {
      flushPara();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i]!)) body.push(lines[i++]!);
      blocks.push({ kind: "code", text: body.join("\n") });
      continue;
    }

    if (line.trim() === "") {
      flushPara();
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      flushPara();
      blocks.push({ kind: "heading", level: heading[1]!.length, text: heading[2]! });
      continue;
    }

    if (UL.test(line)) {
      flushPara();
      const items: string[] = [];
      while (i < lines.length && UL.test(lines[i]!)) items.push(UL.exec(lines[i++]!)![1]!);
      i--;
      blocks.push({ kind: "ul", items });
      continue;
    }

    if (OL.test(line)) {
      flushPara();
      const items: string[] = [];
      while (i < lines.length && OL.test(lines[i]!)) items.push(OL.exec(lines[i++]!)![1]!);
      i--;
      blocks.push({ kind: "ol", items });
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flushPara();
      const body: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i]!)) {
        body.push(lines[i++]!.replace(/^\s*>\s?/, ""));
      }
      i--;
      blocks.push({ kind: "quote", text: body.join("\n") });
      continue;
    }

    para.push(line);
  }
  flushPara();
  return blocks;
}

const HEADING_CLASS: Record<number, string> = {
  1: "text-base font-semibold",
  2: "text-base font-semibold",
  3: "text-sm font-semibold",
};

export function Markdown({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className="space-y-2 leading-relaxed">
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "heading":
            return (
              <p key={i} className={HEADING_CLASS[b.level] ?? "text-sm font-semibold"}>
                {renderInline(b.text, `h${i}`)}
              </p>
            );
          case "ul":
            return (
              <ul key={i} className="list-disc space-y-1 pl-5">
                {b.items.map((it, j) => (
                  <li key={j}>{withBreaks(it, `ul${i}-${j}`)}</li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={i} className="list-decimal space-y-1 pl-5">
                {b.items.map((it, j) => (
                  <li key={j}>{withBreaks(it, `ol${i}-${j}`)}</li>
                ))}
              </ol>
            );
          case "code":
            return (
              <pre
                key={i}
                className="overflow-x-auto rounded bg-neutral-900 p-2.5 font-mono text-xs text-neutral-100"
              >
                <code>{b.text}</code>
              </pre>
            );
          case "quote":
            return (
              <blockquote
                key={i}
                className="border-l-2 border-neutral-300 pl-3 text-neutral-600"
              >
                {withBreaks(b.text, `q${i}`)}
              </blockquote>
            );
          default:
            return <p key={i}>{withBreaks(b.text, `p${i}`)}</p>;
        }
      })}
    </div>
  );
}
