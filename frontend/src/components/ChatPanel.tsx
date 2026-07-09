// Advisory chat panel (web-app-plan §13). Talks to the run-scoped agent at
// POST /api/runs/{runId}/chat, which answers as a Server-Sent Events stream.
//
// EventSource only does GET, so we stream the POST response body ourselves:
// read the ReadableStream, split on SSE frame boundaries ("\n\n"), and parse
// each `data:` line. Event shapes from the backend (app/chat.py):
//   {type:"session", session_id}  — sent first; reused for multi-turn continuity
//   {type:"tool", name}           — agent invoked a query tool (shown as a chip)
//   {type:"token", content}       — a chunk of the assistant's answer
//   {type:"error", message}
//   {type:"done"}
import { useRef, useState } from "react";

type Role = "user" | "assistant";
type ChatMsg = { role: Role; content: string; tools?: string[] };

type StreamEvent =
  | { type: "session"; session_id: string }
  | { type: "tool"; name: string }
  | { type: "token"; content: string }
  | { type: "error"; message: string }
  | { type: "done" };

// A few starter prompts so the panel is useful without knowing what to ask.
const SUGGESTIONS = [
  "Which criteria failed, and why?",
  "What are the top 3 things to change in this video?",
  "How could we fix the failed criteria?",
];

const TOOL_LABELS: Record<string, string> = {
  query_parser_results: "read ABCD verdicts",
  query_scene_descriptions: "read scenes",
  query_visual_objects: "read objects",
  query_audio_segments: "read audio",
  query_text_events: "read on-screen text",
  query_global_stats: "read pacing stats",
  query_field_descriptions: "read field docs",
};

export function ChatPanel({ runId }: { runId: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    // Defer to after the DOM updates.
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    });
  }

  // Mutate the last (assistant) message in place during streaming.
  function updateLastAssistant(fn: (m: ChatMsg) => ChatMsg) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = fn(copy[copy.length - 1]!);
      return copy;
    });
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || streaming) return;
    setError(null);
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: message },
      { role: "assistant", content: "", tools: [] },
    ]);
    setStreaming(true);
    scrollToBottom();

    try {
      const resp = await fetch(`/api/runs/${runId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId.current }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`Server returned ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep: number;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const dataLine = frame
            .split("\n")
            .find((l) => l.startsWith("data:"));
          if (!dataLine) continue;
          const evt = JSON.parse(dataLine.slice(5).trim()) as StreamEvent;
          handleEvent(evt);
        }
      }
    } catch (e) {
      setError((e as Error).message);
      updateLastAssistant((m) => ({
        ...m,
        content: m.content || "(no response)",
      }));
    } finally {
      setStreaming(false);
      scrollToBottom();
    }
  }

  function handleEvent(evt: StreamEvent) {
    switch (evt.type) {
      case "session":
        sessionId.current = evt.session_id;
        break;
      case "tool":
        updateLastAssistant((m) => ({
          ...m,
          tools: m.tools?.includes(evt.name)
            ? m.tools
            : [...(m.tools ?? []), evt.name],
        }));
        break;
      case "token":
        updateLastAssistant((m) => ({ ...m, content: m.content + evt.content }));
        scrollToBottom();
        break;
      case "error":
        setError(evt.message);
        break;
      case "done":
        break;
    }
  }

  return (
    <div className="flex flex-col">
      <div
        ref={scrollRef}
        className="max-h-96 space-y-3 overflow-y-auto rounded border bg-neutral-50 p-3"
      >
        {messages.length === 0 && (
          <div className="space-y-2 text-sm text-neutral-500">
            <p>
              Ask about this video — the agent knows every ABCD verdict and the
              extracted scenes, objects, audio, and on-screen text.
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => void send(s)}
                  className="rounded-full border bg-white px-3 py-1 text-xs text-neutral-700 hover:border-blue-400 hover:bg-blue-50"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
          >
            <div
              className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                m.role === "user"
                  ? "bg-blue-600 text-white"
                  : "border bg-white text-neutral-800"
              }`}
            >
              {m.tools && m.tools.length > 0 && (
                <div className="mb-1 flex flex-wrap gap-1">
                  {m.tools.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-neutral-100 px-1.5 py-0.5 text-[10px] text-neutral-500"
                    >
                      {TOOL_LABELS[t] ?? t}
                    </span>
                  ))}
                </div>
              )}
              <div className="whitespace-pre-wrap">
                {m.content}
                {m.role === "assistant" &&
                  streaming &&
                  i === messages.length - 1 && (
                    <span className="ml-0.5 inline-block animate-pulse">▋</span>
                  )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="mt-3 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
          placeholder="Ask about this video…"
          className="flex-1 rounded border px-3 py-2 text-sm disabled:bg-neutral-100"
        />
        <button
          type="submit"
          disabled={streaming || !input.trim()}
          className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
