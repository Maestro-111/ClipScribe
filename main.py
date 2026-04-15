from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "DODGE_lA2DSd8Ik3Y - Sept 2023 Dodge Hornet.mp4"
video_path = "input/DODGE_lA2DSd8Ik3Y - Sept 2023 Dodge Hornet.mp4"
video_type = "car ad"

clib_scribe_device = "cpu"

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
