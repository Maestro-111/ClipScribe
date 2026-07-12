# Module: Output Parsing & Evaluation

## Purpose
Evaluates extracted video metadata against platform-specific feature criteria. Uses LangGraph ReAct agents backed by an LLM to query the configured SQLAlchemy database and produce structured pass/fail evaluations per feature.

## Key Files
* `parser_core.py`: `VideoInformationParser` — main entry point that resolves the platform evaluator, emits parse-phase progress events, persists per-criterion results to `parser_results`, and writes CSV reports.
* `evaluator_base.py`: `BaseEvaluator` — abstract base class for platform evaluators. Handles baseline vs agentic dispatch, parallel agent execution, and feature-type-to-time-scope resolution.
* `agent.py`: `build_agent` / `run_agent` — builds a LangGraph ReAct agent and runs it with a system prompt, tools, and an optional `time_scope` restriction.
* `advisory.py`: Builds the post-run advisory chat agent over a single run.
* `tools.py`: `build_tools` — constructs LangGraph tool closures (`query_audio_segments`, `query_text_events`, `query_visual_objects`, `query_scene_descriptions`, `query_global_stats`, `query_parser_results`) scoped to a run and grouped by feature type.
* `models.py`: Pydantic models shared across platforms (`BaseFeatureResult`, `BaseAgentEvaluation`).
* `youtube/`: YouTube-specific evaluator, criteria definitions, baseline logic, models, and CSV report generation.

## Guidelines
* The Extractor outputs pure factual data. The Parser's job is to apply business logic or evaluation criteria to that data.
* When adding a new platform, subclass `BaseEvaluator`, define feature configs, and register it in `parser_core.py`.
* All DB query tools live in `tools.py`; add new query tools there and include them in the relevant `tool_map` groups.
* Keep parser progress reporting behind `backend/src/utils/progress.py`; do not import Redis or FastAPI from parser code.
