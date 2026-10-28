import { useState, type ReactNode } from "react";
import type { ExplorationResponse } from "../api/hooks";
import { Pagination } from "./ui";

// Exploratory run report (the exploratory counterpart of the ABCD criteria
// table): KPI strip from the run metrics, then the LLM report — overview,
// chapters, and per-modality digests. Every timestamp seeks the video.

type Report = NonNullable<ExplorationResponse["report"]>;
type Metrics = Report["metrics"];

function clock(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function pct(fraction: number | null): string {
  return fraction === null ? "—" : `${Math.round(fraction * 100)}%`;
}

function SeekButton({
  sec,
  onSeek,
  children,
}: {
  sec: number;
  onSeek: (t: number) => void;
  children?: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={() => onSeek(sec)}
      className="rounded border border-blue-200 bg-white px-1.5 py-0.5 font-mono text-xs text-blue-600 hover:border-blue-400 hover:bg-blue-50"
    >
      {children ?? clock(sec)}
    </button>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-neutral-500">{children}</p>;
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2">
      <div className="text-xs uppercase tracking-wide text-neutral-500">
        {label}
      </div>
      <div className="text-lg font-semibold text-neutral-900">{value}</div>
    </div>
  );
}

function KpiStrip({ m }: { m: Metrics }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      <Kpi label="Duration" value={clock(m.duration_sec)} />
      <Kpi label="Shots" value={String(m.shot_count)} />
      <Kpi label="Cuts / min" value={m.cuts_per_minute.toFixed(1)} />
      <Kpi label="Speech" value={pct(m.speech_coverage)} />
      <Kpi label="On-screen text" value={pct(m.text_coverage)} />
      <Kpi label="Faces" value={m.objects_tracked ? pct(m.face_coverage) : "—"} />
      <Kpi
        label="Words / min"
        value={m.words_per_minute === null ? "—" : String(Math.round(m.words_per_minute))}
      />
    </div>
  );
}

function Chapters({
  chapters,
  onSeek,
}: {
  chapters: Report["chapters"];
  onSeek: (t: number) => void;
}) {
  if (!chapters.length) return <Empty>No chapters were generated.</Empty>;
  return (
    <ol className="space-y-2">
      {chapters.map((c, i) => (
        <li
          key={`${c.start_sec}-${i}`}
          className="rounded-lg border border-neutral-200 p-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <SeekButton sec={c.start_sec} onSeek={onSeek}>
              {clock(c.start_sec)}–{clock(c.end_sec)}
            </SeekButton>
            <span className="font-medium">{c.title}</span>
          </div>
          <p className="mt-1 text-sm text-neutral-600">{c.summary}</p>
        </li>
      ))}
    </ol>
  );
}

function TranscriptTab({
  digest,
  onSeek,
}: {
  digest: Report["transcript"];
  onSeek: (t: number) => void;
}) {
  if (!digest) return <Empty>No speech was detected in this video.</Empty>;
  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-700">{digest.summary}</p>
      {digest.topics.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {digest.topics.map((t) => (
            <span
              key={t}
              className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs text-neutral-700"
            >
              {t}
            </span>
          ))}
        </div>
      )}
      {digest.key_claims.length > 0 && (
        <ul className="space-y-1.5">
          {digest.key_claims.map((c, i) => (
            <li key={`${c.start_sec}-${i}`} className="flex gap-2 text-sm">
              <SeekButton sec={c.start_sec} onSeek={onSeek} />
              <span>{c.text}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Runs with a lot of on-screen text or tracked objects can have dozens of
// rows; page them client-side (the report is already loaded whole).
const PAGE_SIZE = 8;

function usePaging(total: number) {
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  // Clamp in case a different run's (shorter) report arrives on a later page.
  const safePage = Math.min(page, pageCount - 1);
  return { page: safePage, pageCount, start: safePage * PAGE_SIZE, setPage };
}

function PagerFooter({
  paging,
  total,
  noun,
}: {
  paging: ReturnType<typeof usePaging>;
  total: number;
  noun: string;
}) {
  if (total <= PAGE_SIZE) return null;
  const { page, pageCount, start, setPage } = paging;
  return (
    <Pagination
      canPrev={page > 0}
      canNext={page < pageCount - 1}
      onPrev={() => setPage(Math.max(0, page - 1))}
      onNext={() => setPage(Math.min(pageCount - 1, page + 1))}
      label={`${noun} ${start + 1}–${Math.min(start + PAGE_SIZE, total)} of ${total}`}
    />
  );
}

type TextCategory = NonNullable<
  NonNullable<Report["on_screen_text"]>["cards"][number]["category"]
>;

// Keyed by the generated category union, so adding a category in the backend
// enum fails the typecheck here until it has a style.
const CATEGORY_STYLE: Record<TextCategory, string> = {
  brand: "bg-indigo-50 text-indigo-700",
  product: "bg-violet-50 text-violet-700",
  slogan: "bg-fuchsia-50 text-fuchsia-700",
  cta: "bg-green-50 text-green-700",
  price: "bg-amber-50 text-amber-700",
  promo: "bg-orange-50 text-orange-700",
  url: "bg-blue-50 text-blue-700",
  social: "bg-sky-50 text-sky-700",
  legal: "bg-stone-100 text-stone-700",
  caption: "bg-neutral-100 text-neutral-700",
  other: "bg-neutral-100 text-neutral-500",
};

function OnScreenTextTab({
  digest,
  onSeek,
}: {
  digest: Report["on_screen_text"];
  onSeek: (t: number) => void;
}) {
  const cards = digest?.cards ?? [];
  const paging = usePaging(cards.length);
  if (!digest) return <Empty>No on-screen text was detected.</Empty>;

  const { start } = paging;
  const visibleCards = cards.slice(start, start + PAGE_SIZE);

  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-700">{digest.summary}</p>
      <ul className="space-y-2">
        {visibleCards.map((c, i) => (
          <li key={`${c.text}-${start + i}`} className="rounded-lg border border-neutral-200 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{c.clean_text}</span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${CATEGORY_STYLE[c.category]}`}
              >
                {c.category}
              </span>
              <span className="text-xs text-neutral-400">{c.total_sec}s on screen</span>
            </div>
            {c.clean_text !== c.text && (
              <p className="mt-0.5 text-xs text-neutral-400">OCR read: {c.text}</p>
            )}
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {c.spans.map((s) => (
                <SeekButton key={s.start_sec} sec={s.start_sec} onSeek={onSeek}>
                  {clock(s.start_sec)}–{clock(s.end_sec)}
                </SeekButton>
              ))}
            </div>
          </li>
        ))}
      </ul>
      <PagerFooter paging={paging} total={cards.length} noun="Text" />
    </div>
  );
}

const PROMINENCE_STYLE: Record<string, string> = {
  high: "bg-teal-50 text-teal-700",
  medium: "bg-neutral-100 text-neutral-700",
  low: "bg-neutral-100 text-neutral-500",
};

function VisualsTab({
  digest,
  metrics,
  onSeek,
}: {
  digest: Report["visuals"];
  metrics: Metrics;
  onSeek: (t: number) => void;
}) {
  const subjects = digest?.subjects ?? [];
  const paging = usePaging(subjects.length);
  if (!digest) return <Empty>No visual data for this run.</Empty>;

  const visibleSubjects = subjects.slice(paging.start, paging.start + PAGE_SIZE);
  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-700">{digest.summary}</p>
      {!metrics.objects_tracked ? (
        <Empty>Object tracking was skipped for this run.</Empty>
      ) : subjects.length === 0 ? (
        <Empty>No tracked objects.</Empty>
      ) : (
        <>
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
            <tr>
              <th className="py-1 font-medium">Subject</th>
              <th className="py-1 font-medium">Prominence</th>
              <th className="py-1 font-medium">Screen time</th>
              <th className="py-1 font-medium">First seen</th>
            </tr>
          </thead>
          <tbody>
            {visibleSubjects.map((s) => (
              <tr key={s.label} className="border-t border-neutral-100">
                <td className="py-1.5 font-medium">{s.label}</td>
                <td className="py-1.5">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${PROMINENCE_STYLE[s.prominence]}`}
                  >
                    {s.prominence}
                  </span>
                </td>
                <td className="py-1.5 text-neutral-600">{s.screen_time_sec.toFixed(1)}s</td>
                <td className="py-1.5">
                  <SeekButton sec={s.first_seen_sec} onSeek={onSeek} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <PagerFooter paging={paging} total={subjects.length} noun="Objects" />
        </>
      )}
    </div>
  );
}

const TABS = ["Transcript", "On-screen text", "Visuals"] as const;
type Tab = (typeof TABS)[number];

export function ExplorationPanel({
  data,
  onSeek,
}: {
  data: ExplorationResponse;
  onSeek: (t: number) => void;
}) {
  const [tab, setTab] = useState<Tab>("Transcript");
  const report = data.report;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 font-medium">Overview</h2>
        {report ? (
          <>
            <KpiStrip m={report.metrics} />
            <p className="mt-4 text-sm leading-relaxed text-neutral-700">
              {report.summary}
            </p>
          </>
        ) : (
          <p className="text-sm text-amber-700">
            No exploration report was written for this run — the analysis step
            may have failed.
          </p>
        )}
      </section>

      {report && (
        <>
          <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 font-medium">Chapters</h2>
            <Chapters chapters={report.chapters} onSeek={onSeek} />
          </section>

          <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex gap-1 border-b border-neutral-200">
              {TABS.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  aria-pressed={tab === t}
                  className={`-mb-px border-b-2 px-3 py-1.5 text-sm font-medium ${
                    tab === t
                      ? "border-blue-600 text-blue-700"
                      : "border-transparent text-neutral-500 hover:text-neutral-800"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            {tab === "Transcript" && (
              <TranscriptTab digest={report.transcript} onSeek={onSeek} />
            )}
            {tab === "On-screen text" && (
              <OnScreenTextTab digest={report.on_screen_text} onSeek={onSeek} />
            )}
            {tab === "Visuals" && (
              <VisualsTab
                digest={report.visuals}
                metrics={report.metrics}
                onSeek={onSeek}
              />
            )}
          </section>

          {(report.findings ?? []).length > 0 && (
            <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
              <h2 className="mb-3 font-medium">Findings</h2>
              <ul className="space-y-2">
                {(report.findings ?? []).map((f, i) => (
                  <li key={i} className="rounded-lg border border-neutral-200 p-3 text-sm">
                    <p>{f.observation}</p>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {f.evidence_secs.map((s) => (
                        <SeekButton key={s} sec={s} onSeek={onSeek} />
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
