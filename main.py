from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "CHRYSLER_jtuJbB1QXd8 - Nov 2024 Pacifica.mp4"
video_path = "input/CHRYSLER_jtuJbB1QXd8 - Nov 2024 Pacifica.mp4"
video_type = "car ad"

clib_scribe_device = "cpu"

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
