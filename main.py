from src.clip_scribe.build_clip_scribe import build_clip_scribe

video_name = "RAM_KSAA1CYCoEw - 2022 Ram 1500 Classic ｜ Ram 4x4 Winter Event.mp4"
video_path = "input/RAM_KSAA1CYCoEw - 2022 Ram 1500 Classic ｜ Ram 4x4 Winter Event.mp4"
video_type = "car ad"
clib_scribe_device = "cpu"

clib_scribe = build_clip_scribe(video_name, video_path, video_type, clib_scribe_device)

clib_scribe.run()
