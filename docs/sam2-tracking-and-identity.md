# SAM2 Tracking & Cross-Shot Identity in the Extractor

> Scope: this document explains the object-tracking and identity-resolution
> machinery in `backend/src/extractor/extractor_core.py`
> (`VideoInformationExtractor`). It focuses on the SAM2 propagation loop,
> per-frame metadata recording (`_save_metadata`), DINOv2 re-ID embedding
> accumulation (`_extract_embedding`), and cross-shot identity merging
> (`_resolve_identities`). It deliberately treats SAM2 and GroundingDINO as
> black boxes (per the repo's third-party rules) and describes only the
> *contract* the extractor relies on.

---

## 1. Where this sits in the pipeline

For each **shot** (a scene-cut segment from `_digest_video`), the extractor:

1. Samples frames, asks GPT for a scene description + DINO prompt.
2. Generates/resolves a per-shot taxonomy.
3. Walks the shot in **chunks**, alternating *detection* and *SAM2
   propagation* (the focus of this doc).
4. After all shots, resolves cross-shot identities and writes the per-run
   summary under `artifacts/<run_id>/`.

The tracking layer answers two questions:

- **Within a shot:** where is each detected object, frame by frame? → SAM2
  mask propagation.
- **Across shots:** is the car in shot 5 the *same* car as in shot 2? →
  DINOv2 embedding similarity in `_resolve_identities`.

---

## 2. Core state

| Field | Type | Lifecycle | Purpose |
|---|---|---|---|
| `obj_id_counter` | `int` | **Whole run**, never reset | Hands out globally-unique local IDs (`_get_next_obj_id`). |
| `active_trackers` | `dict[int, TrackerData]` | **Reset every shot** (`= {}`) | The extractor's mirror of "which objects are on screen right now." Drives `_is_new_object`. |
| `id_to_label` | `dict[int, str]` | **Whole run**, never reset | Maps each local ID → its resolved semantic label. Pure bookkeeping. |
| `object_registry` | `dict[int, ObjectRegistryEntry]` | **Whole run**, entries never deleted | Per-object accumulator: boxes, timestamps, embedding sum/count. Feeds the final summary. |
| `inference_state` | SAM2 opaque state | **Reset every shot** (`reset_state`) | SAM2's internal memory of all prompts added this shot. |

Two asymmetries are the source of almost every subtlety below:

- **`active_trackers` resets per shot; `obj_id_counter` and `object_registry`
  do not.** IDs are unique across the whole video; the registry keeps every
  object ever seen.
- **`active_trackers` can forget an object mid-shot; SAM2 cannot.** Once an ID
  is added via `add_new_points_or_box`, SAM2 keeps it until `reset_state` at
  the next shot boundary.

---

## 3. The chunked detect → propagate loop

Detection (GroundingDINO + MTCNN) is **expensive**, so it does **not** run on
every frame. Instead each shot is processed in **chunks** of at most
`detection_interval` frames (default 10):

```
one chunk = [ detect @ frame D ]  ──►  [ SAM2 propagate D → D+k ]

shot:
  chunk 1:  detect@x1 ─ propagate x1→x2
  chunk 2:  detect@x3 ─ propagate x3→x4     (x3 = x2 + 1)
  chunk 3:  detect@x5 ─ propagate x5→x6     (x5 = x4 + 1)
  ...until current_frame >= end_f
```

Per chunk, in `extract`:

1. Read frame at `current_frame` (= `D`).
2. **Detect**: DINO objects + MTCNN faces.
3. For each detection, resolve a taxonomy label, then `_is_new_object`:
   - If it does **not** overlap (IoU > 0.5, same/related label) an entry in
     `active_trackers` → `_add_new_tracker`: allocate a new ID, store it in
     `id_to_label` + `active_trackers`, and register a SAM2 prompt
     (`add_new_points_or_box`) at frame `D`.
4. If `active_trackers` is empty, skip ahead `frames_to_track` and `continue`.
5. **Propagate**: `propagate_in_video(start_frame_idx=D,
   max_frame_num_to_track=k)` yields one `(frame_idx, obj_ids,
   video_res_masks)` tuple per frame in `[D, D+k]`.
6. For each yielded frame → `_save_metadata` + `visualize_sam_tracking`.
7. `current_frame = last_propagated_frame + 1`.

### What the generator actually yields

```python
for frame_idx, obj_ids, video_res_masks in chunk_generator:
    ...
```

- `frame_idx` — the absolute frame being reported.
- `obj_ids` — **every object ID SAM2 is currently tracking this shot**, *not*
  "objects detected in this frame." Detection only ran once, at `D`.
- `video_res_masks` — one mask **logit map** per `obj_id`, at video
  resolution. A tracked object that is *not visible* in `frame_idx` still
  appears in `obj_ids`, just with an **empty** mask.

This distinction — "always in `obj_ids`, but the mask may be empty" — is what
the next section turns into per-frame decisions.

---

## 4. `_save_metadata`: turning masks into records

Called once per propagated frame. It records OCR text, appends raw detection
rows for the run-inspector overlay, and, per object, stores a bounding box +
optional embedding.

### 4.1 Text (the easy half)

```python
masks_np = masks.cpu().numpy()
...
for cur in frame_text:
    if self._is_valid_text(cur, h, max_text_height):
        self.text_registry[second_key].add(cur["text"])
```

Note: `frame_text` (OCR) was computed **once per chunk** at `D` and is
**replayed** for every propagated frame. Because `text_registry` is a `set`
keyed by integer second, this dedupes naturally. OCR is therefore *not* re-run
per propagated frame — a deliberate cost-saving approximation.

### 4.2 Objects

```python
for i, obj_id in enumerate(obj_ids):
    mask_binary = masks_np[i] > 0.0          # (A)
    current_box = self._mask_to_box(mask_binary)

    if current_box:                          # (B)
        ... record box, maybe accumulate embedding ...
        self.active_trackers[obj_id] = {"box": current_box, "label": label}
    else:                                    # (C)
        self.active_trackers.pop(obj_id, None)
```

**(A) `masks_np[i] > 0.0`** — `video_res_masks` are raw mask **logits**.
Thresholding at `0.0` is equivalent to `sigmoid(logit) > 0.5`, i.e.
"probability of foreground > 50%". The result is a boolean array marking that
object's pixels.

**(B) the `if current_box` check** — `_mask_to_box` returns `None` when the
mask is empty (`np.where(mask > 0)` finds no pixels). An empty mask means SAM2
believes the object is **not visible this frame** (occluded / off-screen /
lost). So:

- **box present** → object on screen → append box + timestamp to the registry,
  periodically accumulate an embedding, and refresh `active_trackers`.
- **box absent (C)** → object gone → **pop from `active_trackers`**. The
  `object_registry` entry is **not** deleted; it simply stops growing.

The pop is the *only* way `active_trackers` shrinks, and it is what lets a
later detection believe a reappeared object is "new" (see §6).

### 4.3 Embedding accumulation (re-ID fuel)

Every `reid_model_frame_check_freq` frames (currently 10 in `clip_scribe.yaml`;
the constructor fallback is 20 if the key is omitted), the object's crop is
embedded and *maybe* folded into a running mean:

```python
new_emb = self._extract_embedding(current_frame_img, current_box)

if embedding_count == 0:
    embedding_sum += new_emb; embedding_count = 1          # first sample: always take
else:
    current_mean = embedding_sum / embedding_count          # a VECTOR (384,), not a scalar
    cos_sim = dot(new_emb, current_mean) / (||new_emb|| * ||current_mean||)
    if cos_sim < 0.85:                                      # only keep NOVEL viewpoints
        embedding_sum += new_emb; embedding_count += 1
```

This is a **multi-view accumulator**. It only folds in an embedding when the
new view is *different enough* (`cos_sim < 0.85`) from what's already averaged,
so near-duplicate frames don't over-weight whichever angle happened to appear
most. The accumulated mean becomes the object's signature for cross-shot
matching.

`current_mean` is a **vector** of shape `(384,)` (DINOv2 ViT-S CLS token), not
a scalar — `embedding_sum` is `np.zeros(384)` divided by an `int`.

> **Config note:** this novelty threshold is exposed as
> `reid_similarity_difference` under `clip_scribe.extractor` in
> `clip_scribe.yaml` (current value `0.8`; code default `0.8`). It is read in
> `build_clip_scribe.build_extractor` and passed to the extractor as
> `self.reid_similarity_difference`, alongside the other re-ID knobs
> (`word_similarity_threshold`, `label_match_merge_threshold`,
> `label_no_match_merge_threshold`). Note there remain *other* hardcoded
> magic numbers in this file — the `0.5` IoU in `_is_new_object` and the
> `0.85` `wup_similarity` cutoff in `_labels_match` — which are still
> candidates for naming.

### 4.4 `_extract_embedding` tensor handling

```python
img_tensor = self.embedding_transform(crop_rgb).unsqueeze(0).to(self.dino_reid_device)
with torch.no_grad():
    features = self.reid_model.forward_features(img_tensor)
    embedding = features["x_norm_clstoken"]
return embedding.cpu().numpy().flatten()
```

- `embedding_transform(crop_rgb)` → torchvision transforms (resize / normalize
  / ToTensor) producing a `(C, H, W)` tensor.
- `.unsqueeze(0)` → adds a batch dim → `(1, C, H, W)`; `forward_features`
  expects a batch, not a bare image.
- `.to(self.dino_reid_device)` → the input must live on the **same device** as
  the re-ID model or the forward pass throws a device-mismatch error.
- `.cpu().numpy().flatten()` → NumPy cannot read GPU/MPS memory, so the result
  must come back to CPU before `.numpy()`. Everything downstream
  (`embedding_sum`, cosine sims) is NumPy, so CPU is the right home; this also
  frees the GPU tensor.

---

## 5. Identity within a shot: the lifecycle states

The hard part. An object can disappear and reappear, and *what ID it ends up
with* depends on **timing** relative to the chunk boundaries and on whether
SAM2 re-acquires it.

### The two timescales

```
"inside a propagation"      "across a checkpoint"
(within one chunk's          (a new chunk's detect step
 generator loop)              has run in between)

 detect@D                     detect@D ── prop ──┐
   │                                             │ detect@D'
   └─ prop: f0 f1 f2 f3 ...                       └─ prop ...
        ▲ no detection here                  ▲ NEW detection here
```

- **No detection runs *inside* a propagation.** So a flicker inside one chunk
  cannot create a new ID.
- **A new ID can only be born at a `detect@D` step**, and only if the object
  is *absent from `active_trackers`* at that moment.

### Decisive fact

> SAM2 **always keeps the old ID registered** (it is in `obj_ids` every frame),
> but "registered" ≠ "emits a non-empty mask." Whether the old ID **re-acquires**
> (mask becomes non-empty again) when the object returns is an **empirical
> property of SAM2's tracker**, not a guarantee.

### State SINGLE — recovered inside one propagation (or before the next detect)

The object vanishes and returns **within the same chunk**, *or* SAM2
re-acquires the old ID **before** the next `detect@D`. Because no detection
sees it as absent, **no new ID is created**:

```
ID 5:  ●●●●● .... ●●●●●        one ID, a gap in its timestamp list
        └ vanish ┘ └ regained by SAM2, re-added to active_trackers
```

`_save_metadata` finds the existing `object_registry[5]` entry and simply
appends. **One local ID, one global ID.** Most common for short occlusions.

### State MERGED — new ID born, old ID stays dead, lifespans disjoint

The object is absent across a `detect@D`, so a **new ID** is born. SAM2 then
**never re-acquires** the old ID (long gap / appearance drift). The two
lifespans do **not** overlap:

```
ID 5:  ●●●●●                   lifespan x1..x2
ID 9:              ●●●●●        lifespan x5..x6
            ↑ disjoint in time
```

`_resolve_identities` later compares their mean embeddings and, if similar,
**merges them into one global ID.** *Two local IDs, one global ID.* This is the
intended recovery path — and the *primary* mechanism **across shots** (where
`reset_state` guarantees the old ID is gone).

### State DUPLICATE — new ID born, old ID re-acquires, lifespans overlap

The object is absent across a `detect@D` (→ new ID born), and **then** SAM2
re-acquires the old ID while the new ID is also live. Both emit masks for the
same frames:

```
ID 5:  ●●●●●        ●●●●●       re-acquired at x5..x6
ID 9:              ●●●●●        x5..x6
                   ↑ both active at the SAME frames → OVERLAP
```

Overlapping lifespans trip the guard in `_resolve_identities`
(`is_overlapping → continue`) **before** any embedding comparison, so the two
**never merge.** *One physical object → two global IDs (a phantom duplicate).*

### Summary table

| Old ID after reappearance | New ID born? | Lifespans | Outcome |
|---|---|---|---|
| Re-acquired *within* the same chunk, or *before* next detect | No | one ID (gap) | **SINGLE** |
| Stays dead permanently | Yes | disjoint | **MERGED** (1 global) |
| Re-acquired *after* the new ID is born | Yes | overlap | **DUPLICATE** (2 globals) |

> **Frequency is unmeasured.** SINGLE and DUPLICATE are expected to dominate;
> in-shot MERGED is the tail case. The exact split depends on SAM2's
> re-acquisition behavior and chunk timing, which we have not instrumented. To
> measure it, log per-frame `(obj_id, mask_empty?)` and watch whether a
> vanished ID ever returns non-empty — this requires a real extraction run.

---

## 6. Why a new ID gets created: `_is_new_object`

```python
def _is_new_object(self, new_box, new_label) -> bool:
    for obj_id, tracker_data in self.active_trackers.items():   # <-- only active_trackers
        if iou(new_box, tracker_data["box"]) > 0.5 and labels_match(...):
            return False
    return True
```

The decision consults **only `active_trackers`** — *not* SAM2's current mask
set. So once an object has been popped (its mask went empty), a re-detection at
the next `detect@D` cannot see that SAM2 still tracks it, and a new ID is
minted. This is the root cause of State DUPLICATE.

```
        active_trackers  │  SAM2 knows
                         │
 detect@D5:  ID 5 popped │  ID 5 still registered  ── _is_new_object can't see this
             → "new!"    │
             → ID 9 born │  now SAM2 tracks BOTH 5 and 9
```

---

## 7. `_resolve_identities`: cross-shot merging (and its blind spot)

Anchored, greedy, ordered by first appearance:

```python
object_ids = sorted(registry, key=lambda k: registry[k]["timestamps"][0])

for i, id_a in enumerate(object_ids):
    if id_a in id_map: continue
    assign id_a a fresh global id
    for id_b in object_ids[i+1:]:
        if id_b in id_map: continue
        if overlap(a, b): continue                # <-- the guard
        if embeddings_similar(a, b):              # threshold logic below
            id_map[id_b] = global_id_of(a)
            end_a = max(end_a, end_b)             # <-- window EXTENSION
```

Merge decision (after the overlap guard, both must have `embedding_count > 0`):

- `labels_match AND visual_sim > label_match_merge_threshold`, **or**
- `visual_sim > label_no_match_merge_threshold` (visual override: merge even if
  labels differ, when they look very alike).

Two behaviors matter:

1. **The overlap guard** (`max(start_a,start_b) < min(end_a,end_b)`) encodes a
   correct physical assumption: *one object cannot be in two places at the same
   instant.* If two tracks are live simultaneously, they must be different
   objects → skip. This is right 99% of the time — and it is exactly why it
   **cannot repair State DUPLICATE**, whose two tracks overlap by construction.

2. **Window extension** — when an anchor absorbs a match, it extends its own
   `end_a`. This has a consequence for duplicates (next section).

### Why the duplicate survives every downstream stage

Suppose an earlier instance of the same object exists (e.g. ID 2 from a
previous shot), disjoint from both duplicates ID 5 and ID 9:

```
ID 2:  ●●●●                         (earlier shot)
ID 5:        ●●●●●     ●●●●●         (duplicate A, earlier first-appearance)
ID 9:                  ●●●●●         (duplicate B)
```

1. ID 2 is the anchor (earliest). It scans forward, reaches **ID 5** (disjoint,
   embeddings match) → **merges ID 5**, and **extends its window** to cover
   ID 5's end (now reaching into ID 9's interval).
2. ID 2 continues to **ID 9** — but ID 2's window now overlaps ID 9 →
   `is_overlapping` True → **ID 9 skipped.**
3. ID 9 is never absorbed → later becomes **its own global ID.**

So the earlier object absorbs **one** duplicate (deterministically, the first
that clears the threshold — normally the earlier-starting one), and the
window-extension **strands the other** as a phantom global. If ID 5 fails the
threshold but ID 9 passes, it flips — but the result is identical in shape:
**one absorbed, one phantom.**

```
Result:   global G0 = {ID 2, ID 5}      ← real, merged
          global G1 = {ID 9}            ← PHANTOM (same physical object, counted again)
```

**Key takeaway:** a duplicate, once created, propagates all the way to the
output. `_resolve_identities` can never make the count correct — it fixes
*temporal fragmentation* (the same object across time), never *concurrent
duplication* (the same object tracked twice at once). The only effective fix is
**upstream** — preventing the second ID from being born.

---

## 8. From registry to summary: `_finalize_data` & `_calculate_metrics`

`_finalize_data`:

1. Calls `_resolve_identities` → `local_id → global_id` map.
2. For each registry entry, computes motion/spatial metrics via
   `_calculate_metrics` (velocity, growth, screen coverage, direction,
   centrality, screen-time ratio, quadrant) from the boxes + timestamps.
3. Groups occurrences under their global ID into `VisualObjectSummary`.
4. Emits text events (from `text_registry`), plus the audio and scene-
   description registries.

Because each phantom duplicate is a distinct global ID, it shows up as an extra
`visual_object` with its own occurrence + metrics — i.e. the duplicate inflates
object counts and aggregate screen-time in the final JSON / DB.

---

## 9. Known limitations & recommendations

1. **Concurrent duplicates are unrepairable downstream.** The merge stage
   cannot dissolve a same-object pair with overlapping lifespans. Mitigate
   **upstream**: have `_is_new_object` also check against SAM2's *current*
   non-empty masks (not just `active_trackers`), or de-duplicate overlapping
   same-label tracks before `_resolve_identities`.

2. **OCR is replayed per chunk, not re-run per frame.** Acceptable for
   second-level text events; document it so consumers don't assume per-frame
   OCR fidelity.

3. **Remaining hardcoded thresholds.** The re-ID viewpoint-novelty threshold is
   now configurable (`reid_similarity_difference`), but `0.5` (IoU in
   `_is_new_object`), `0.85` (`wup_similarity` in `_labels_match`), and the OCR
   `ignore_terms` list are still inline. Promote the behavioral ones to
   `clip_scribe.yaml` for consistency with the existing re-ID thresholds.

4. **Greedy, anchor-first merging is not best-match.** `_resolve_identities`
   merges the *first* candidate over threshold, not the most similar. Combined
   with window extension, this determines *which* duplicate is absorbed but
   never removes the phantom. A best-match (e.g. Hungarian / highest-similarity)
   pass would be more robust if duplicates persist.

5. **State frequencies are unmeasured.** Add a cheap per-frame
   `(obj_id, mask_empty?)` debug log to quantify how often SINGLE / MERGED /
   DUPLICATE actually fire before investing in a fix.

---

## Appendix: quick reference

| Symbol in code | Meaning |
|---|---|
| `masks_np[i] > 0.0` | foreground mask for object `i` (logit > 0 ≈ prob > 0.5) |
| `current_box is None` | object not visible this frame → pop from `active_trackers` |
| `embedding_sum / embedding_count` | mean re-ID embedding **vector** `(384,)` |
| `cos_sim < 0.85` | new viewpoint novel enough to accumulate |
| `is_overlapping` | two tracks live at the same time → assumed distinct → never merged |
| `end_a = max(end_a, end_b)` | anchor window extension on merge (strands later duplicates) |
