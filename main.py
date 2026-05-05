from src.clip_scribe.build_clip_scribe import build_clip_scribe
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

video_name = "JEEP_EvQO3sH1SMs - 2023 Jeep Grand Cherokee L ｜ Jeep No Limits.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"
clib_scribe_device = "mps"

# Optional: provide explicit hints, otherwise they are auto-generated from the video name
# user_hints = ["brand logo", "truck"]

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
