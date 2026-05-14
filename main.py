from src.clip_scribe.build_clip_scribe import build_clip_scribe
from src.clip_scribe.build_clip_scribe_plalform import build_platform
from dotenv import load_dotenv, find_dotenv

import sys
import traceback

load_dotenv(find_dotenv())

video_name = "CHRYSLER_jtuJbB1QXd8 - Nov 2024 Pacifica.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"


run_id = ""

clib_scribe_device = "mps"
clib_scribe_platform = "youtube"
clib_scribe_mode = "full"


platform_params = {
    "youtube_brand_name": "Chrysler",
    "youtube_branded_products": ["Chrysler Pacifica"],
    "youtube_branded_products_categories": [
        "Chrysler Pacifica minivan",
        "Chrysler Pacifica grand caravan",
    ],
    "youtube_call_to_actions": [
        "learn more",
        "buy now",
        "order now",
        "purchase now",
        "get now",
        "apply now",
    ],
}

# Optional: provide explicit hints for extractor, otherwise they are auto-generated from the video name
user_hints = ["suv", "vehicle", "car", "truck", "license plate", "brand logo"]


try:
    clib_scribe_platform_conf = build_platform(clib_scribe_platform, **platform_params)

    if clib_scribe_platform_conf is not None:
        clib_scribe = build_clip_scribe(
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            clib_scribe_mode=clib_scribe_mode,
            clib_scribe_device=clib_scribe_device,
            clib_scribe_platform_name=clib_scribe_platform,
            user_hints=user_hints,
            clib_scribe_platform_conf=clib_scribe_platform_conf,
        )

        print(clib_scribe)
        clib_scribe.run(run_id=run_id)

    else:
        raise NotImplementedError(f"no {clib_scribe_platform} available")
except Exception:
    traceback.print_exc()
    sys.exit(1)
