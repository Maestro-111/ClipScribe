from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "RAM_f7XWwCEaTdM - Nov 2024 RamDS.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"

clib_scribe_device = "cpu"

# Optional: provide explicit hints, otherwise they are auto-generated from the video name
user_hints = ["brand logo", "truck"]

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
