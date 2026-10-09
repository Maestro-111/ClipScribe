"""Contract coverage for the async writer's shared operations."""

import asyncio
import types
from pathlib import Path

from clip_scribe.db import (
    ClipScribeAsyncReaderDB,
    ClipScribeAsyncWriterDB,
    create_async_db_engine,
)
from clip_scribe.db.schema import metadata_obj


def test_async_writer_methods_share_sync_write_operations(tmp_path: Path) -> None:
    asyncio.run(_exercise_async_writer(tmp_path / "async-writer.db"))


async def _exercise_async_writer(database_path: Path) -> None:
    engine = create_async_db_engine(f"sqlite:///{database_path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata_obj.create_all)

        writer = ClipScribeAsyncWriterDB(async_engine=engine)
        reader = ClipScribeAsyncReaderDB(async_engine=engine)

        run_id = await writer.save_run(
            run_id="run-1",
            video_name="video.mp4",
            video_path="videos/video.mp4",
            video_type="advertisement",
            video_metadata={
                "global_stats": {"total_shots": 1},
                "visual_objects": [
                    {
                        "global_id": 1,
                        "label": "car",
                        "occurrences": [
                            {
                                "shot_index": 0,
                                "lifespan": [0.0, 1.0],
                                "screen_coverage": 0.5,
                            }
                        ],
                    }
                ],
                "text_events": [{"second": 0, "text": ["ClipScribe"]}],
                "audio_segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "scene_descriptions": [
                    {
                        "shot_index": 0,
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "description": "A car",
                    }
                ],
                "shot_boundaries": [
                    {"index": 0, "start": 0.0, "end": 1.0, "duration": 1.0}
                ],
                "frame_detections": [
                    {
                        "shot_index": 0,
                        "frame_idx": 0,
                        "timestamp_sec": 0.5,
                        "source": "dino",
                        "label": "car",
                    }
                ],
            },
            field_descriptions={"runs": {"video_name": "Uploaded filename"}},
        )
        assert run_id == "run-1"
        assert (await reader.get_run(run_id))["video_name"] == "video.mp4"
        assert len(await reader.get_visual_objects(run_id)) == 1
        assert len(await reader.get_text_events(run_id)) == 1
        assert len(await reader.get_audio_segments(run_id)) == 1
        assert len(await reader.get_scene_descriptions(run_id)) == 1
        assert len(await reader.get_shot_boundaries(run_id)) == 1
        assert len(await reader.get_frame_detections(run_id)) == 1
        assert len(await reader.get_field_descriptions("runs")) == 1

        await writer.insert_video(
            user_id="user-1",
            content_hash="hash-1",
            stored_key="video-key.mp4",
            original_name="video.mp4",
            size_bytes=42,
        )
        await writer.touch_video("user-1", "hash-1")
        video = await reader.get_video_by_hash("user-1", "hash-1")
        assert video is not None
        assert video["stored_key"] == "video-key.mp4"

        await writer.create_job(
            job_id="job-1",
            mode="full",
            run_id=run_id,
            status="queued",
            created_by="user-1",
        )
        await writer.update_job(
            "job-1",
            status="running",
            started_at="2026-10-02T10:00:00+00:00",
        )
        assert (await reader.get_job("job-1"))["status"] == "running"
        assert await writer.update_job_if_status(
            "job-1",
            allowed_statuses=("running",),
            status="completed",
        )
        assert not await writer.update_job_if_status(
            "job-1",
            allowed_statuses=("queued",),
            status="failed",
        )
        assert await writer.reset_job_for_retry("job-1", run_id="run-2")
        reset_job = await reader.get_job("job-1")
        assert reset_job["status"] == "queued"
        assert reset_job["run_id"] == "run-2"

        await writer.add_chat_message(
            run_id=run_id,
            session_id="run-session",
            role="user",
            content="Run question",
        )
        await writer.add_chat_message(
            job_id="job-1",
            session_id="job-session",
            role="assistant",
            content="Job answer",
            tool_calls=["get_runs"],
        )
        assert await writer.delete_chat_session(run_id, "run-session") == 1
        assert await writer.delete_job_chat_session("job-1", "job-session") == 1

        parser_result = types.SimpleNamespace(
            platform="youtube",
            feature_name="logo",
            evaluation=True,
        )
        await writer.save_parser_results(run_id, "youtube", [parser_result])
        assert len(await reader.get_parser_results(run_id)) == 1

        assert await writer.delete_job("job-1")
        assert await reader.get_job("job-1") is None
        await writer.delete_run(run_id)
        assert await reader.get_run(run_id) is None
        assert await reader.get_parser_results(run_id) == []

        await writer.delete_video("user-1", "video-key.mp4")
        assert await reader.list_videos("user-1") == []
    finally:
        await engine.dispose()
