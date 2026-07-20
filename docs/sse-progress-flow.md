# Live Progress (SSE) — End-to-End Flow

This doc visualizes how a running job's progress travels from the Python
pipeline, through Redis, out the FastAPI SSE endpoint, and into the React live
page. Read it alongside the code — every box names the file that owns it.

The whole feature exists to answer one question in the browser: *"what is this
job doing right now?"* — with the twist that the answer must also work for a user
who opens the page **mid-run** or **just after it finished**.

---

## 1. The one-paragraph mental model

The pipeline never talks to the browser. It only calls `reporter.emit(...)`.
In the web process that reporter writes each event as one entry into a **Redis
Stream** named `job:{id}:stream`. The browser opens an `EventSource` to
`GET /jobs/{id}/events`; that endpoint **replays the whole stream from the
beginning, then blocks and tails it for new entries**, forwarding each one to the
browser as an SSE `data:` frame. A React `useReducer` folds those frames into the
live view. A Redis **Stream** (not pub/sub) is the key choice: it stores history,
so a late subscriber still gets everything.

```
 PIPELINE  ──emit()──►  REDIS STREAM  ──XRANGE+XREAD──►  SSE ENDPOINT  ──data:──►  BROWSER
 (torch)               job:{id}:stream                 (slim API)              (React reducer)
```

---

## 2. The players (who owns what)

| Layer | File | Role |
|-------|------|------|
| Event vocabulary | `backend/src/utils/progress.py` | `ProgressEvent.*` string constants + the abstract `ProgressReporter` the core depends on. Torch-free, web-free. |
| Emit sites | `backend/src/clip_scribe/engine.py`, `backend/src/extractor/extractor_core.py` | Call `self.progress.emit(...)` at fixed pipeline points. |
| Web sink | `backend/app/events.py` | `RedisProgressReporter.emit` → `XADD` into the stream. Also the log bridge + cancel token. |
| Wiring | `backend/app/job_execution.py` (`run_job_core`) | Builds the Redis reporter and injects it into the engine for the duration of the job. |
| SSE endpoint | `backend/app/routes/jobs.py` (`_job_event_stream`, `job_events`) | Replays the stream, then tails it; renders each entry as an SSE frame. |
| Browser transport | `frontend/src/routes/jobs.$jobId.tsx` (`LeafJob`, `useEffect`) | `new EventSource(...)`; dispatches each frame into the reducer. |
| Browser state | same file (`reducer`, `LiveState`) | Folds events into phases / current shot / logs / progress %. |

Notice the **dependency direction**: the core (`src/`) knows only the abstract
`ProgressReporter`. Redis, FastAPI, and SSE live in `app/`. The CLI and tests
inject `NullProgressReporter`, which drops every event — the pipeline runs
identically with or without a live tail.

---

## 3. The event vocabulary

These are the only `type` values that ever flow through the stream. The Python
constant, the wire string, and the frontend reducer `case` are all the same
literal — that's the contract that keeps the two languages in sync.

```
job.started       ── carries `phases: [...]` + video_name  (seeds the phase tree)
  phase.started   ── {phase, total_shots?}
    shot.started            {shot_idx}
    shot.scene_described    {description, dino_prompt}
    shot.taxonomy_resolved  {targets: [...]}
    shot.frame_processed    (increments a counter)
    shot.completed
  phase.completed  ── {phase, total_shots?}
  audio.segment    ── {start, end, text}
  log              ── {level, message}   (injected by the log bridge, not emit sites)
job.completed     ── {run_id}     ⟵ TERMINAL
job.failed        ── {error}      ⟵ TERMINAL
job.canceled                      ⟵ TERMINAL
```

`log` is special: it is **not** produced by an `emit()` call site. It is produced
by `JobLogStreamHandler`, a `logging.Handler` attached to the `clip_scribe`
logger. Any `logger.info(...)` anywhere in the pipeline is mirrored into the
current job's stream — see §6.

---

## 4. Producer side — how an event becomes a stream entry

```
┌─ backend/src/extractor/extractor_core.py ───────────────────────────┐
│  self.progress.emit(ProgressEvent.SHOT_STARTED, {"shot_idx": i})    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  self.progress is whatever run_job_core injected
                               ▼
┌─ backend/app/events.py :: RedisProgressReporter.emit ───────────────┐
│  entry = {"type": "shot.started", "data": '{"shot_idx": 3}'}        │  ← flat string fields
│  client.xadd("job:{id}:stream", entry, maxlen=STREAM_MAXLEN, ~)     │  ← one entry appended
│                                                                     │
│  if type == "job.started":  SET job:{id}:started = entry (TTL)      │  ← durable snapshot, see §7
│  if type in TERMINAL_EVENTS: EXPIRE the stream (TTL)                │  ← stream self-cleans
│  (every Redis call wrapped in try/except — emit MUST NOT raise)     │
└─────────────────────────────────────────────────────────────────────┘
```

Key properties:

- **A stream entry is flat string fields.** Redis stream values are strings, so
  the payload is JSON-encoded into a single `data` field (`_entry()`). The
  endpoint reverses this in `_sse_frame()`.
- **`maxlen` trimming is approximate.** Long, log-heavy runs evict the oldest
  entries — including the early `job.started`. That's the whole reason for the
  `job:{id}:started` snapshot key (§7).
- **emit never raises.** A Redis hiccup logs a warning and is swallowed;
  progress is best-effort and must never break a running job.

### Where the reporter comes from (`run_job_core`)

```
┌─ backend/app/job_execution.py :: run_job_core ──────────────────────┐
│  reporter  = make_reporter(redis_url, job_id)   # Redis, or Null    │
│  canceller = make_canceller(redis_url, job_id)                      │
│  install_job_log_bridge(redis_url)              # attach log handler │
│  token = current_job_id.set(job_id)             # tag logs w/ job id │
│  engine = builder.build_clip_scribe(..., progress_reporter=reporter)│
│  engine.run(run_id=run_id)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

Both execution paths (inline API executor and Celery worker) call this same
function, so live progress works identically no matter how the job was
dispatched. If Redis is down, `make_reporter` returns `NullProgressReporter`
and the job still runs — just with no live tail.

---

## 5. Consumer side — the SSE endpoint (the clever part)

`GET /jobs/{id}/events` → `_job_event_stream` is an async generator. It has two
phases: **replay**, then **tail**.

```
┌─ backend/app/routes/jobs.py :: _job_event_stream ───────────────────────────┐
│                                                                             │
│  key = "job:{id}:stream";  last_id = "0"                                    │
│                                                                             │
│  ── PHASE A: REPLAY ────────────────────────────────────────────────────   │
│  entries = XRANGE key            # everything currently in the stream       │
│  if no job.started in entries:   # it aged out via maxlen trimming          │
│      snapshot = GET job:{id}:started                                        │
│      if snapshot: yield it first   # re-seed the phase tree                 │
│  for (id, fields) in entries:                                               │
│      last_id = id                                                           │
│      yield  "data: {type, data}\n\n"                                        │
│      if type in TERMINAL_EVENTS: return   # finished job → replay is done   │
│                                                                             │
│  if job row status is terminal: return   # backstop, e.g. canceled queued  │
│                                                                             │
│  ── PHASE B: TAIL ──────────────────────────────────────────────────────   │
│  while True:                                                                │
│      resp = XREAD {key: last_id} BLOCK 15s COUNT 100   # block for new ones │
│      if timeout / no data:                                                  │
│          yield  ": keepalive\n\n"        # SSE comment, keeps socket warm   │
│          if job row terminal: return     # backstop for no terminal event  │
│          continue                                                          │
│      for (id, fields) in resp:                                             │
│          last_id = id                                                       │
│          yield  "data: {type, data}\n\n"                                    │
│          if type in TERMINAL_EVENTS: return   # normal end of stream        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Why each piece exists:

- **`last_id = "0"` then `XRANGE`** → a client that connects mid-run, or minutes
  after the job finished (while the stream is still within its TTL), gets the
  **full history** before any live update. This is the core reason a Stream is
  used instead of pub/sub.
- **`XREAD ... BLOCK 15s`** → the server holds the request open, cheaply, until a
  new entry appears or 15s pass. On timeout it emits a `: keepalive` comment so
  proxies don't kill an idle connection. `socket_timeout` is set above the block
  window so the client socket doesn't time out before the blocking read returns.
- **Terminal event → `return`** → the generator ends, FastAPI closes the HTTP
  response, and the browser's `EventSource` sees the close.
- **The `job_is_terminal()` DB backstop** → covers jobs that never emit a
  terminal event (a queued job canceled before it ran, or Redis being down at run
  time). Without it the tail would block forever.

`job_events` itself is thin: 404 if the job row is missing, then wrap the
generator in a `StreamingResponse(media_type="text/event-stream")` with
`Cache-Control: no-cache` and `X-Accel-Buffering: no` (disables nginx buffering
so frames arrive immediately).

### One entry, three representations

```
Redis entry (flat strings):   {"type": "shot.started", "data": "{\"shot_idx\": 3}"}
        │  _sse_frame() parses `data`, re-nests, JSON-encodes the whole thing
        ▼
SSE wire frame:               data: {"type":"shot.started","data":{"shot_idx":3}}\n\n
        │  EventSource splits on \n\n → e.data is the text after "data: "
        ▼
Browser Event object:         { type: "shot.started", data: { shot_idx: 3 } }
```

---

## 6. The log bridge (a parallel producer)

Logs reach the browser through a completely separate path from `emit()`:

```
logger.info("...")  anywhere in the clip_scribe pipeline
        │
        ▼
┌─ backend/app/events.py :: JobLogStreamHandler.emit ─────────────────┐
│  job_id = current_job_id.get()      # ContextVar set by run_job_core │
│  if job_id is None: return          # ambient startup logs dropped   │
│  XADD job:{id}:stream {type:"log", data:{level, message}}           │
└─────────────────────────────────────────────────────────────────────┘
```

`install_job_log_bridge` attaches this handler to the `clip_scribe` logger once
per process (idempotent). The `current_job_id` ContextVar is how it knows which
job's stream to write to **without threading a job id through every logging call
site**. Set at the top of `run_job_core`, reset in its `finally`.

So the stream interleaves two producers: structured `emit()` events and mirrored
log lines. The endpoint forwards both; the reducer routes `log` frames to the log
tail and everything else to structured state.

---

## 7. The two "late subscriber" edge cases (why the extra machinery exists)

Most of the non-obvious code exists to make the page correct for someone who
**wasn't watching from the start**.

**Case 1 — connect mid-run, `job.started` already trimmed.**
`job.started` carries the phase list that builds the whole phase tree. It's the
first entry, so approximate `maxlen` trimming evicts it first on a chatty run.
Fix: the reporter also writes a standalone `job:{id}:started` key (untrimmed,
TTL'd). On replay, if the endpoint doesn't see `job.started` in the stream, it
re-emits that snapshot first. Both sides also **grow the phase list from
`phase.started`/`phase.completed` events** as a second fallback (see the reducer
cases and `summarize_progress`), so the tree renders even if both signals are
gone.

**Case 2 — connect after the job finished.**
The stream is TTL'd (set on the terminal event), so for a while after completion
the replay still contains the full history ending in `job.completed`. The client
gets everything and then the stream closes immediately. After the TTL expires,
the stream is gone; the live view falls back to the job row's status (fetched via
`useJob`), and the "View run →" link uses the persisted `run_id`.

**Retry reuses the job id.** `reset_stream` deletes the stream, cancel flag, and
started snapshot so a reconnecting client doesn't replay the *previous* run's
terminal event and close instantly.

---

## 8. Browser side — EventSource → reducer → UI

```
┌─ frontend/src/routes/jobs.$jobId.tsx :: LeafJob ────────────────────┐
│  useEffect(() => {                                                   │
│    const es = new EventSource(`/api/jobs/${jobId}/events`)          │
│    es.onopen    = () => dispatch({type:"@stream/open"})   // "live"  │
│    es.onmessage = (e) => dispatch(JSON.parse(e.data))     // event   │
│    es.onerror   = () => { es.close(); dispatch(@stream/close) }     │
│    return () => es.close()                // cleanup on unmount      │
│  }, [jobId])                                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ reducer(state, action) — folds each event into LiveState ──────────┐
│  job.started            → phaseOrder, phases={p:"pending"}, videoName│
│  phase.started          → phases[p]="running", totalShots (+grow)    │
│  phase.completed        → phases[p]="completed" (+grow phaseOrder)   │
│  shot.started           → currentShot = {idx, framesDone:0}          │
│  shot.scene_described    → currentShot.description / dinoPrompt      │
│  shot.taxonomy_resolved  → currentShot.targets                      │
│  shot.frame_processed    → currentShot.framesDone++                  │
│  shot.completed          → shotsCompleted++                         │
│  audio.segment           → append to audioSegments                   │
│  log                     → append to logs (capped at 500)           │
│  job.completed           → runId, streamStatus="closed"             │
│  job.failed              → error,  streamStatus="closed"            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
        UI: progress bar (overallProgress), phase tree, current-shot
            card, live log tail, "live/connecting/closed" indicator
```

Notes:

- **`es.onerror` is the normal end.** When the server returns after a terminal
  event, `EventSource` surfaces the close as an error. The handler calls
  `es.close()` so the browser doesn't auto-reconnect forever, and marks the
  stream closed.
- **`overallProgress`** mirrors the server's `_PHASE_WEIGHT` weighting exactly
  (`scene_detection 0.05, audio 0.15, shot_processing 0.7, finalize 0.1,
  parse 0.3`), normalized over whatever phases this mode actually ran.
- **The reducer is idempotent-ish on reconnect**: because replay re-sends the
  whole history, a fresh `EventSource` rebuilds state from scratch — which is why
  `useEffect` is keyed on `[jobId]` and tears down the old stream on change.

---

## 9. Two consumers of the same stream

The jobs **list** page can't open one SSE connection per row, so it uses a
cheaper second endpoint:

```
GET /jobs/{id}/events    → SSE, full live detail        (one open job page)
GET /jobs/{id}/progress  → one-shot % , poll per row    (jobs list / batch table)
```

`job_progress` does a single `XRANGE` and reduces it with `summarize_progress`
(`backend/app/events.py`) — the **same phase weighting** as the frontend — to
return a coarse `percent`. Completed jobs report 100 without touching Redis. In
the UI, `ChildRow` polls it via `useJobProgress(child.job_id, running)` to draw
the small per-run bar in a batch. Same stream, two read patterns: tail for
detail, one-shot reduce for a list bar.

---

## 10. End-to-end trace of a single event

Following `shot.started` for shot 3, front to back:

```
1. extractor_core.py         self.progress.emit("shot.started", {"shot_idx": 3})
2. RedisProgressReporter     XADD job:{id}:stream {type:"shot.started", data:'{"shot_idx":3}'}
3. (event sits in the stream; also survives for late replay)
4. _job_event_stream TAIL    XREAD returns the new entry
5. _sse_frame                → 'data: {"type":"shot.started","data":{"shot_idx":3}}\n\n'
6. StreamingResponse         flushes the frame down the open HTTP connection
7. EventSource.onmessage     e.data = '{"type":"shot.started","data":{"shot_idx":3}}'
8. dispatch(JSON.parse(...)) reducer case "shot.started"
9. state.currentShot = {idx: 3, framesDone: 0}
10. React re-renders the "Current shot" card → "Shot 3 · 0 frames"
```

That's the whole pipe. Everything else in this doc is making that pipe correct
for late subscribers, crash-safe (emit never raises), and self-cleaning (TTLs).
