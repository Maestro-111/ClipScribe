from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "RAM_QT1IQtE62Uk - Ram 1500 Classic DS - Hockey - QBC FR - Jan.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"

clib_scribe_device = "cpu"

# Optional: provide explicit hints, otherwise they are auto-generated from the video name
# user_hints = ["hockey stick", "hockey puck", "ice rink", "helmet"]

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
