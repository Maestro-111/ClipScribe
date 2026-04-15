from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "JEEP_EvQO3sH1SMs - 2023 Jeep Grand Cherokee L ｜ Jeep No Limits.mp4"
video_path = "input/JEEP_EvQO3sH1SMs - 2023 Jeep Grand Cherokee L ｜ Jeep No Limits.mp4"
video_type = "car ad"

clib_scribe_device = "cpu"

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
