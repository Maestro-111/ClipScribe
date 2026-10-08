"""Contract coverage for the async reader's shared operations."""

import asyncio
import json
from pathlib import Path

from src.db import ClipScribeAsyncReaderDB, create_async_db_engine
from src.db.schema import (
    audio_segments_table,
    chat_messages_table,
    field_descriptions_table,
    frame_detections_table,
    global_stats_table,
    jobs_table,
    metadata_obj,
    parser_results_table,
    runs_table,
    scene_descriptions_table,
    shot_boundaries_table,
    text_events_table,
    videos_table,
    visual_object_occurrences_table,
)


def test_async_reader_methods_share_sync_query_operations(tmp_path: Path) -> None:
    asyncio.run(_exercise_async_reader(tmp_path / "async-reader.db"))


async def _exercise_async_reader(database_path: Path) -> None:
    engine = create_async_db_engine(f"sqlite:///{database_path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata_obj.create_all)
            await conn.execute(
                runs_table.insert(),
                {"run_id": "run-1", "video_name": "video.mp4"},
            )
            await conn.execute(
                global_stats_table.insert(),
                {
                    "run_id": "run-1",
                    "total_shots": 1,
                    "qp_intro_shots": json.dumps([0]),
                    "qp_general_rapid_fire_segments": json.dumps([[0, 1]]),
                },
            )
            await conn.execute(
                audio_segments_table.insert(),
                {"run_id": "run-1", "start_time": 0.25, "text": "hello"},
            )
            await conn.execute(
                text_events_table.insert(),
                {"run_id": "run-1", "second": 0, "line_index": 0, "text": "RAM"},
            )
            await conn.execute(
                visual_object_occurrences_table.insert(),
                {
                    "run_id": "run-1",
                    "global_id": 1,
                    "label": "car",
                    "lifespan_start": 0.0,
                },
            )
            await conn.execute(
                field_descriptions_table.insert(),
                {
                    "table_name": "runs",
                    "column_name": "video_name",
                    "description": "Uploaded filename",
                },
            )
            await conn.execute(
                scene_descriptions_table.insert(),
                {
                    "run_id": "run-1",
                    "shot_index": 0,
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "description": "A car",
                },
            )
            await conn.execute(
                shot_boundaries_table.insert(),
                {
                    "run_id": "run-1",
                    "shot_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                },
            )
            await conn.execute(
                frame_detections_table.insert(),
                {
                    "run_id": "run-1",
                    "frame_idx": 0,
                    "timestamp_sec": 0.5,
                    "source": "dino",
                },
            )
            await conn.execute(
                parser_results_table.insert(),
                {
                    "run_id": "run-1",
                    "platform": "youtube",
                    "feature_name": "logo",
                    "evaluation": True,
                },
            )
            await conn.execute(
                jobs_table.insert(),
                [
                    {
                        "job_id": "parent-1",
                        "parent_job_id": None,
                        "run_id": None,
                        "status": "queued",
                        "created_by": "user-1",
                        "params_json": None,
                    },
                    {
                        "job_id": "child-1",
                        "parent_job_id": "parent-1",
                        "run_id": "run-1",
                        "status": "completed",
                        "created_by": "user-1",
                        "params_json": {"platform": "youtube"},
                    },
                ],
            )
            await conn.execute(
                chat_messages_table.insert(),
                [
                    {
                        "run_id": "run-1",
                        "job_id": None,
                        "session_id": "run-session",
                        "role": "assistant",
                        "content": "Run answer",
                        "tool_calls_json": ["get_scenes"],
                    },
                    {
                        "run_id": None,
                        "job_id": "parent-1",
                        "session_id": "job-session",
                        "role": "user",
                        "content": "Job question",
                        "tool_calls_json": None,
                    },
                ],
            )
            await conn.execute(
                videos_table.insert(),
                {
                    "user_id": "user-1",
                    "content_hash": "hash-1",
                    "stored_key": "video-key.mp4",
                    "original_name": "video.mp4",
                },
            )

        reader = ClipScribeAsyncReaderDB(async_engine=engine)

        assert (await reader.get_run("run-1"))["video_name"] == "video.mp4"
        assert (await reader.get_latest_run())["run_id"] == "run-1"
        assert (await reader.get_global_stats("run-1"))["qp_intro_shots"] == [0]
        assert len(await reader.get_audio_segments("run-1", 1.0)) == 1
        assert len(await reader.get_text_events("run-1", 1)) == 1
        assert len(await reader.get_visual_objects("run-1", "car", 1.0)) == 1
        assert len(await reader.get_field_descriptions("runs")) == 1
        assert len(await reader.get_scene_descriptions("run-1", 1.0, 2.0)) == 1
        assert len(await reader.list_runs()) == 1
        assert len(await reader.get_shot_boundaries("run-1")) == 1
        assert len(await reader.get_frame_detections("run-1", 0.0, 1.0)) == 1
        assert len(await reader.get_parser_results("run-1")) == 1

        run_messages = await reader.get_chat_messages("run-1", "run-session")
        assert run_messages[0]["tool_calls_json"] == ["get_scenes"]
        assert len(await reader.get_job_chat_messages("parent-1", "job-session")) == 1
        assert len(await reader.get_chat_sessions("run-1")) == 1
        assert len(await reader.get_job_chat_sessions("parent-1")) == 1

        assert (await reader.get_job("child-1"))["params_json"] == {
            "platform": "youtube"
        }
        assert len(await reader.list_jobs("user-1")) == 2
        assert len(await reader.list_parent_jobs("user-1", status="completed")) == 1
        assert len(await reader.get_child_jobs("parent-1")) == 1
        assert len(await reader.get_run_siblings("run-1", "user-1")) == 1
        assert await reader.run_owner("run-1") == "user-1"

        assert (await reader.get_video_by_hash("user-1", "hash-1"))[
            "stored_key"
        ] == "video-key.mp4"
        assert (await reader.get_video_by_key("user-1", "video-key.mp4"))[
            "content_hash"
        ] == "hash-1"
        assert len(await reader.list_videos("user-1")) == 1
    finally:
        await engine.dispose()
