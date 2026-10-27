# Module: Output Parsing & Evaluation

## Purpose
Evaluates extracted video metadata against platform-specific feature criteria. Uses LangGraph ReAct agents backed by an LLM to query the configured SQLAlchemy database and produce structured pass/fail evaluations per feature.

## Key Files
* `parser_core.py`: `VideoInformationParser` — main entry point that resolves the platform evaluator, emits parse-phase progress events, persists per-criterion ABCD results to `abcd_results`, and writes CSV reports.
* `evaluator_base.py`: `BaseEvaluator` — abstract base class for platform evaluators. Handles baseline vs agentic dispatch, parallel agent execution, and feature-type-to-time-scope resolution.
* `../agent_factory.py`: `AgentBuilder` builds LangGraph ReAct agents (ABCD/exploration run chat, ABCD job chat); `AgentRunner.run_abcd_agent` runs one with a system prompt, tools, and an optional `time_scope` restriction.
* `prompts.py`: System prompts for the ABCD run and job advisory chat agents.
* `tools.py`: `build_tools` constructs run-scoped LangGraph tool closures (`query_audio_segments`, `query_text_events`, `query_visual_objects`, `query_scene_descriptions`, `query_global_stats`, `query_parser_results`) grouped by feature type; `build_job_tools` constructs the job-scoped tool set (`list_job_runs`, `query_job_scorecard`, cross-run `query_parser_results`, plus per-run detail tools).
* `models.py`: Pydantic models shared across platforms (`BaseFeatureResult`, `BaseAgentEvaluation`).
* `explore/`: Exploratory analysis (`parsing_mode=exploratory`). `metrics.py` computes deterministic run metrics, deduplicated OCR text cards, and per-shot context from `ExtractionSummary` (no LLM); `explorer.py` (`VideoExplorer`) runs parallel structured-output digest calls plus an overview call and resolves the LLM's index references back to real timestamps; `models.py` holds `ExploreReport`, persisted to `exploration_results`.
* `youtube/`: YouTube-specific evaluator, criteria definitions, baseline logic, models, and CSV report generation.

## Guidelines
* The Extractor outputs pure factual data. The Parser's job is to apply business logic or evaluation criteria to that data.
* When adding a new platform, subclass `BaseEvaluator`, define feature configs, and register it in `parser_core.py`.
* Run-scoped extraction query tools shared with the exploration agent live in `../shared_tools.py` (`build_run_data_tools`); ABCD-specific ones (`query_parser_results`, the job tools) live in `tools.py`. Add new tools in the matching place and include them in the relevant `tool_map` groups.
* Keep parser progress reporting behind `backend/src/utils/progress.py`; do not import Redis or FastAPI from parser code.
