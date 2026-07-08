// Hand-maintained mirror of the Pydantic run-inspector models in
// backend/app/models.py. Replace these with imports from the auto-generated
// types.ts once `pnpm gen:api` is re-run after the backend models land.

export interface Run {
  run_id: string;
  video_name: string | null;
  video_path: string | null;
  video_type: string | null;
  created_at: string | null;
}

export type DetectionSource = "dino" | "ocr" | "mtcnn" | "sam_mask";

export interface FrameDetection {
  id: number;
  run_id: string;
  shot_index: number | null;
  frame_idx: number | null;
  timestamp_sec: number | null;
  source: DetectionSource | null;
  label: string | null;
  text: string | null;
  box_x1: number | null;
  box_y1: number | null;
  box_x2: number | null;
  box_y2: number | null;
  confidence: number | null;
  object_id: number | null;
}

export interface ShotBoundary {
  id: number;
  run_id: string;
  shot_index: number | null;
  start_sec: number | null;
  end_sec: number | null;
  duration_sec: number | null;
}

export interface GlobalStatsResponse {
  global_stats: Record<string, unknown> | null;
  shot_boundaries: ShotBoundary[];
}

export interface ParserResult {
  id: number;
  run_id: string;
  platform: string | null;
  feature_category: string | null;
  feature_name: string | null;
  feature_criteria: string | null;
  evaluation: boolean | null;
  llm_prompt: string | null;
  llm_explanation: string | null;
  langsmith_run_id: string | null;
  created_at: string | null;
}

export interface AudioSegment {
  id: number;
  run_id: string;
  start_time: number | null;
  end_time: number | null;
  text: string | null;
  confidence: number | null;
}

export interface TextEvent {
  id: number;
  run_id: string;
  second: number | null;
  line_index: number | null;
  text: string | null;
}
