# ClipScribe — Cloud & Kubernetes Deployment Design

A planning doc, not a spec. It picks up where `web-app-plan.md` §12 leaves off
and works through what it actually takes to run ClipScribe **at scale** on GCP:
the Kubernetes nuances (especially GPU scheduling), the managed cloud resources
required, and the decisions we still need to make.

This is the phase where the product outgrows the local MVP. Today one worker
processes one job = one `run_id` = one video, and a busy worker leaves the
backlog in Redis. That model already scales *horizontally by adding workers* —
what it does **not** yet support is (a) one job spanning many videos, (b) video
and artifacts living in object storage instead of a local disk, and (c) elastic
GPU capacity. Those three are the gap between "works on my Mac + one GPU box"
and "a user uploads 20 videos and we process them in parallel."

Sections marked **Choice** are decisions to make before the corresponding piece
is built. Where this doc restates something from `web-app-plan.md`, it links the
section rather than duplicating it.

---

## 1. What already helps (don't rebuild it)

The current backend was written with this split in mind, so a surprising amount
carries over unchanged:

- **API ↔ worker process boundary is real** (`web-app-plan` §8). The slim API
  imports `app/celery_app.py` only and dispatches by task *name*
  (`celery_app.send_task("app.tasks.run_job", …)` in `app/job_runner.py`), so it
  never pulls in torch. `settings.py` loads **no models** in `celery` mode. The
  API is already a stateless, CPU-only, horizontally-scalable process.
- **Queue behavior is already tuned for a fleet.** `app/celery_app.py` sets
  `worker_prefetch_multiplier=1` + `task_acks_late=True`, so a busy worker does
  **not** hoard the backlog — extra work stays in Redis for the next free
  worker/machine. Adding a second worker (or a hundred) needs zero code change.
- **The per-video unit of work is isolated.** `run_job_core` in
  `app/job_execution.py` drives exactly one video to a terminal state and is
  shared by both dispatch paths. Fan-out (below) reuses it verbatim — we call it
  N times, we don't change what it does.
- **Load-once builder** (`web-app-plan` §3). Model loading is amortized across
  every job a worker process handles. This is what makes a *warm* worker pool
  (below) worth it.
- **Artifact-upload seam already exists** (`web-app-plan` §9.2). `backend/src/utils/artifacts.py`
  has an `ArtifactUploader` with a `SimulatedGCSArtifactUploader` that currently
  logs the `gs://…/<run_id>/artifacts.tar.gz` it *would* push. Swapping in a real
  GCS client is a drop-in, no call-site change.

So this doc is mostly about the *deployment substrate* and three application
changes, not a rewrite.

---

## 2. The three MVP-breaking application changes

These are prerequisites; the cloud topology assumes they exist.

### 2.1 Break the `job = one video` coupling (fan-out)

**Today:** `JobService.create_job` (`app/job_runner.py`) mints one `run_id` and
calls `_dispatch` once. **Target:** a job is a *batch* of videos that share
brand/product context, fanned out to one Celery task per video.

```
POST /jobs { videos: [v1, v2, … v20], platform, platform_params }
  → write ONE parent job row
  → write N run rows (one run_id per video)
  → send N Celery tasks  (celery `group`)
  → return the parent job_id; client polls parent + per-video children
```

The per-video task is exactly today's `run_job` — untouched. What changes:

- **Schema.** A parent/child relationship on `jobs`: add `parent_job_id TEXT
  NULL` (self-FK) so a batch is a parent row with N child rows, each keyed to its
  own `run_id`. Alembic migration; schema stays migration-owned (`web-app-plan`
  §4). Reuse the existing `jobs` columns for children.
- **`create_job`.** Loop the `build_task_payload` + `_dispatch` over the video
  list; write the parent row first. Consider Celery `group()` / `chord()` if we
  want a completion callback that flips the parent to a terminal state; otherwise
  derive parent status from children on read.
- **Parent status aggregation.** `completed` when all children terminal;
  `failed`/`partial` if some fail; `canceled` if canceled before children finish.
- **API + polling.** `GET /jobs/{id}` returns the parent plus a per-video child
  summary so the UI can show a 20-row progress panel. The per-job Redis progress
  stream (`web-app-plan` §5/§9) is already keyed by `job_id`, so each child gets
  its own stream for free.

**Choice — batch semantics.** Does a batch share one taxonomy/hint context
across videos, or is each video fully independent (just grouped for UX)? The
"same brand/products" assumption suggests shared `platform_params`, which we
already pass per-job; decide whether we also want cross-video identity/taxonomy
sharing (bigger change, touches `taxonomy_core.py`) or keep videos independent
(recommended for v1).

### 2.2 Move ingest to signed-URL object storage

**Today:** `video_path` is relative to a local `INPUT_DIR`, and
`JobService._resolve_input` does a filesystem traversal guard. That cannot work
when the browser uploads gigabytes and the worker is an ephemeral pod on another
machine.

- Browser requests a **signed PUT URL** from the API, uploads **directly to a
  GCS bucket** — the API never touches the bytes (this is the open item in
  `web-app-plan` §9.1).
- `video_path` becomes a `gs://bucket/uploads/<id>/<name>` URI. `_resolve_input`'s
  local check is replaced by "object exists in the bucket." The traversal guard
  becomes irrelevant.
- The **worker** downloads the object to pod-local scratch (an `emptyDir` /
  ephemeral SSD) at task start, runs, and uploads results back. This is why the
  artifact change (2.3) is coupled to this one.

**Choice — signed URL vs. resumable upload.** Large video files over flaky
connections favor GCS **resumable uploads**; a simple signed PUT is easier.
Recommend resumable for anything user-facing.

### 2.3 Artifacts must land in GCS, not local disk

Worker pods are ephemeral and share no filesystem, so `artifacts/<run_id>/`
(`web-app-plan` §9.15) cannot be the source of truth for the run inspector. Wire
the real GCS uploader behind the existing `remote_artifact_write` flag and make
the artifact-serving routes (`GET /runs/{id}/video`, `/tracked-video`,
`/png/...` in `web-app-plan` §6) read from GCS — either by redirecting to a
signed GET URL (preferred; keeps bytes off the API) or streaming through.

**Choice — per-file objects vs. one tarball.** The current simulated uploader
pushes a single `artifacts.tar.gz`. The inspector wants random access to
individual PNGs and the tracked mp4, which a tarball defeats. Recommend
per-object layout under `gs://…/<run_id>/` so the API can hand out signed GETs
per artifact.

---

## 3. Kubernetes topology

```
                    ┌──────────────┐
   Browser ───────► │  Ingress /   │  (GCLB + managed cert)
   (SPA + signed    │  Gateway     │
    GCS uploads)    └──────┬───────┘
                          │
                    ┌──────▼───────┐        ┌───────────────┐
                    │  API Deploy  │───────►│ Memorystore   │  broker + progress stream
                    │  (CPU, slim, │        │ (Redis)       │
                    │   N replicas)│◄───────┤               │
                    └──────┬───────┘        └───────┬───────┘
                          │                        │ tasks pulled
                          ▼                        ▼
                    ┌──────────────┐        ┌───────────────────────────┐
                    │  Cloud SQL   │◄──────►│  GPU Worker Deployment     │
                    │  (Postgres)  │        │  node pool: GPU            │
                    └──────────────┘        │  1 pod = 1 GPU, warm       │
                                            │  KEDA-scaled on queue depth│
                          ▲                 └──────────────┬────────────┘
                          │                                │
                    ┌──────┴───────┐                        ▼
                    │     GCS      │◄───────────────────────┘
                    │ uploads +    │  video in, artifacts out
                    │ artifacts    │
                    └──────────────┘
```

### 3.1 The API tier

A standard CPU `Deployment` behind a GKE Ingress/Gateway with a managed
certificate. Stateless — scale on CPU/RPS with the HPA. In `celery` mode it
loads only the DB reader/writer and a Redis client, so pods start in seconds and
the image is the slim `backend/docker/api` one (`web-app-plan` §8).

One caveat carried from `web-app-plan` §12: **SSE + timeouts.** `GET
/jobs/{id}/events` is a long-lived stream. Configure the load balancer /
Ingress backend timeout generously (or the stream is cut mid-job). This is the
main reason GKE is cleaner here than Cloud Run for the API, though Cloud Run
remains viable if the SSE timeout is tuned.

### 3.2 The GPU worker tier — the part with real nuance

This is where the deployment earns its complexity. The rules, in order:

**One pod = one GPU.** The worker runs `--pool=solo --concurrency=1` and loads
the ~8–15 GB model stack once into that process's GPU memory. So the pod requests
exactly one GPU:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1     # whole-GPU, non-oversubscribable
```

**A node with N GPUs hosts up to N worker pods.** The NVIDIA device plugin
advertises each physical GPU as one allocatable unit, and the scheduler treats
`nvidia.com/gpu` as a **countable, non-shareable** resource. So an 8-GPU node
runs up to 8 of our worker pods, each pinned to its own GPU. Our Deployment is
just `replicas: K`; the scheduler packs replicas onto multi-GPU nodes until
their GPUs are exhausted, then spills to the next node. The scaling knob is
clean: **total in-flight videos ≈ total GPUs across the pool.**

Things that bite if ignored:

- **The GPU count is a hard cap, not a hint.** Ask for a GPU when none is free
  and the pod stays `Pending` until one frees or the autoscaler adds a GPU node.
  It will *never* cram two pods onto one GPU — which is exactly the
  `concurrency=1` invariant we want, enforced by the scheduler.
- **Don't try two workers per GPU** unless we deliberately opt into GPU sharing
  (time-slicing or MIG on A100/H100). A heavy CV+LLM stack would OOM the card or
  thrash. One job per GPU is correct for us.
- **Set CPU/RAM requests too.** The GPU count caps pods-per-node, but a pod also
  needs CPU + RAM for video decode and model load. Without requests, a pod can be
  GPU-available yet RAM-starved. Size requests so N pods actually fit a node's
  CPU/RAM, not just its GPU count.
- **Node pool must be GPU-tainted; workers tolerate it.** Give the GPU node pool
  a taint (e.g. `nvidia.com/gpu=present:NoSchedule`) so ordinary pods don't land
  on expensive GPU nodes, and put a matching `toleration` + `nodeSelector`
  (accelerator type) on the worker pods only.
- **The NVIDIA driver + device plugin must be installed** on the GPU node pool.
  On GKE this is the managed GPU driver installation DaemonSet — enable it on the
  node pool; don't hand-roll drivers.
- **Keep one GPU type per node pool.** Mixed T4 / L4 / A100 means wildly
  different per-job runtime and memory headroom, which makes `concurrency=1`
  timing and KEDA scaling hard to reason about. One accelerator per pool; add a
  second pool if we need a second GPU class.

### 3.3 Warm pool vs. per-run pods — the decision that matters most

Two ways to give a run its GPU. **Recommend the first, strongly.**

**Recommended — persistent warm worker pool, autoscaled on queue depth.** A GPU
worker `Deployment` stays running with models loaded. **KEDA** scales the replica
count from the Redis queue length (its Redis scaler reads the broker list): 0
queued → scale to a small floor (or zero, if we accept a cold first job); 20
queued → scale up toward the per-GPU cap. The **cluster autoscaler** adds GPU
nodes when replicas go `Pending` and removes them when idle. Workers stay warm,
so the 30–60 s model load is paid *once per pod*, not once per video.

**Not recommended — a fresh GPU pod per run** (K8s `Job`/`Pod` per video). It
pays, *every single time*: image pull (the core image is multi-GB), GPU node
cold-start (often **minutes** if the autoscaler must add a node), plus the 30–60 s
model load. For a 20-video batch that's brutal, and "scale to zero" rarely
recoups it.

> Note the distinction from `web-app-plan` §12 Mode B: the **CLI/batch** mode
> deliberately uses a per-video **Indexed Job** with *no Redis/Celery* because
> it's a one-shot operator-driven run where cold starts are acceptable. The
> **web app** (this doc) wants the warm Celery pool because it's interactive and
> latency-sensitive. Same core image, two launch strategies — pick per mode.

**Choice — scale-to-zero or keep a floor?** Scale-to-zero saves money when idle
but makes the first job of the day eat a full cold GPU-node + model-load start
(minutes). A floor of 1 warm worker keeps latency low at the cost of one idle
GPU. Recommend a floor of 1 during business hours, 0 overnight (KEDA supports a
schedule), revisited once we see real traffic.

### 3.4 Migrations, config, secrets

- **Alembic runs once per release** as a discrete K8s `Job` from the slim API
  image, *not* per replica (`web-app-plan` §4). All replicas share one Cloud SQL
  instance.
- **Config** via `ConfigMap` for non-secret knobs (`CLIPSCRIBE_JOB_BACKEND=celery`,
  `CLIPSCRIBE_DEVICE=cuda`, CORS origins, bucket names) and **GCP Secret Manager**
  (surfaced as env via the CSI driver or synced secrets) for `OPENAI_API_KEY`,
  `LANGCHAIN_*`, and DB creds (`web-app-plan` §8). `CLIPSCRIBE_DEVICE=cuda` on the
  worker pool; the API doesn't care.
- **Cloud SQL access** via the Cloud SQL Auth Proxy sidecar (or Private IP);
  **Redis** via Memorystore private IP on the VPC. Both API and worker need
  network reach to both (`web-app-plan` §3).

---

## 4. Cloud resources (GCP)

| Concern | GCP service | Notes / choice |
|---|---|---|
| Cluster | **GKE** (Standard or Autopilot) | Standard gives explicit GPU node-pool + taint control; Autopilot is less ops but historically more constrained for GPUs. **Choice** below. |
| GPU workers | GKE **GPU node pool** (e.g. L4 / A100) | Tainted, autoscaled, managed NVIDIA driver DaemonSet. GPU class is a cost/perf **choice**. |
| API | GKE CPU node pool (or Autopilot) | Slim image, HPA on CPU/RPS. |
| Broker + progress stream | **Memorystore for Redis** | Managed; both Celery broker and the per-job SSE stream. Don't self-host. |
| Relational DB | **Cloud SQL for PostgreSQL** | Default backend for the web deployment (`web-app-plan` §9.6). Private IP. |
| Video + artifact storage | **Cloud Storage (GCS)** | Signed-URL upload (2.2), artifact objects (2.3). Lifecycle rules for retention (§6). |
| Images | **Artifact Registry** | Slim API image + heavy core image; the core image is 8–15 GB, so co-locate the registry in-region to keep pulls fast. |
| Secrets | **Secret Manager** | LLM keys, DB creds; injected as env at deploy. |
| Ingress / TLS | **GCLB via GKE Ingress/Gateway** + managed cert | Long backend timeout for SSE. |
| Autoscaling glue | **KEDA** (Redis scaler) + cluster autoscaler | Replica count from queue depth; nodes from `Pending` pods (§3.3). |
| GPU quota | **Compute Engine GPU quota** in-region | Real gate on parallelism — request quota early; a 20-way batch needs 20 GPUs of headroom. |
| Observability | Cloud Logging/Monitoring; **LangSmith** already traces parser agents | Add per-job token/cost metrics (`web-app-plan` §9.8). |

**Choice — GKE Standard vs. Autopilot.** Standard gives us direct control over
GPU node pools, taints, and the device plugin, which matches the nuance in §3.2.
Autopilot reduces node ops but has tighter GPU support and less scheduling
control. Recommend **Standard** for the GPU pool given how central GPU packing is
to our cost model; the API tier could live on Autopilot if we want to split.

**Choice — GPU class.** L4 (cheaper, ample for most CV + Whisper inference) vs.
A100 (faster, pricier, MIG-capable). Recommend starting on **L4** and measuring
per-video wall-clock before committing to anything larger.

---

## 5. Cancellation, resiliency, and cost at scale

These are `web-app-plan` open questions that get sharper once many GPUs are in
play.

- **Cooperative cancel becomes important** (`web-app-plan` §9.4 / §10.10).
  Today `cancel_job` avoids `terminate=True` (a hard kill leaks GPU state +
  half-written files), so canceling a running child only suppresses its terminal
  write. At batch scale a user canceling 20 videos wants running ones to stop and
  free their GPUs promptly. Prioritize the `should_cancel` flag in the shot loop
  before this ships — otherwise a canceled batch keeps burning GPU minutes.
- **Worker crash / preemption.** If we use preemptible/Spot GPU nodes to cut
  cost, tasks can die mid-run. `task_acks_late=True` (already set) means an
  un-acked task is redelivered — good — but the engine writes `tracked_output.mp4`
  incrementally, so a crash leaves a partial file. Write to `.partial` and rename
  atomically (`web-app-plan` §9.5), and make the task idempotent per `run_id`.
  **Choice:** Spot GPUs (cheap, interruptible) vs. on-demand (pricier, stable).
- **OpenAI cost is the real per-job spend**, not just GPU time — each video hits
  the LLM for hints, taxonomy, per-shot scene description, and ~30 parser
  criteria. A "submit 20 videos" button multiplies that by 20 with one click.
  A per-user/per-batch budget cap + token accounting should exist *before* this
  is exposed beyond internal use (`web-app-plan` §9.8, §11).
- **Retention.** GCS lifecycle rules on the artifact bucket (delete after N days
  / keep last K) plus a `DELETE /runs/{id}` that clears DB rows and GCS objects
  (`web-app-plan` §9.14). Without this, storage grows unbounded once uploads move
  to the cloud.

---

## 6. Open questions specific to this deployment

Numbered to extend `web-app-plan` §9; these are the ones this doc adds.

1. **Batch context sharing** (2.1) — independent videos grouped for UX, or shared
   taxonomy/identity across a batch? Recommend independent for v1.
2. **Signed PUT vs. resumable upload** (2.2) for large videos on flaky links.
3. **Artifact layout** (2.3) — per-object vs. tarball; API redirect-to-signed-GET
   vs. stream-through.
4. **GKE Standard vs. Autopilot**, and whether to split API (Autopilot) from GPU
   workers (Standard).
5. **GPU class** (L4 vs. A100) and **Spot vs. on-demand** GPU nodes.
6. **Scale-to-zero vs. warm floor** for the worker pool, and the KEDA schedule.
7. **Parent-job completion callback** — Celery `chord` vs. read-time aggregation
   for parent status.
8. **Cost guardrails** — per-batch OpenAI budget cap and where it's enforced
   (API pre-flight vs. worker).
9. **Multi-user / auth** (still `web-app-plan` §9.3) — batches make "whose job is
   this" and per-user quota matter sooner.

---

## 7. Suggested build order

Each step is independently shippable and de-risks the next.

1. **GCS ingest + artifacts** (2.2, 2.3) — flip `remote_artifact_write` to a real
   GCS uploader, add signed-URL upload, make artifact routes serve from GCS. Do
   this first; it's independent of K8s and unblocks ephemeral workers.
2. **Job fan-out** (2.1) — `parent_job_id` migration + `create_job` loop + parent
   status aggregation + UI batch view.
3. **Cooperative cancel** (`web-app-plan` §10.10) — needed before batches are
   exposed, so cancel actually frees GPUs.
4. **Adapt the existing containers for GKE** — publish the slim API image and
   the baked-weight CUDA worker image, then add deployment manifests/release jobs
   around them (`web-app-plan` §8, §10.11).
5. **GKE cluster + managed services** — GPU node pool (tainted, driver DaemonSet),
   Memorystore, Cloud SQL, GCS buckets, Artifact Registry, Secret Manager,
   Ingress. Alembic release Job.
6. **Autoscaling** — KEDA Redis scaler on the worker Deployment + cluster
   autoscaler on the GPU pool; tune floor/ceiling against real traffic.
7. **Cost + retention guardrails** (§5) — token accounting, per-batch budget,
   GCS lifecycle, `DELETE /runs/{id}`.
