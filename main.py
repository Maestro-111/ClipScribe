from src.clip_scribe.build_clip_scribe import build_clip_scribe
from src.clip_scribe.build_clip_scribe_plalform import build_platform
from dotenv import load_dotenv, find_dotenv

import sys
import traceback

load_dotenv(find_dotenv())

video_name = "JEEP_EvQO3sH1SMs - 2023 Jeep Grand Cherokee L ｜ Jeep No Limits.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"
clib_scribe_device = "mps"

clib_scribe_platform = "youtube"

platform_params = {
    "youtube_brand_name": "Jeep",
    "youtube_branded_products": ["Grand Cherokee", "Grand Cherokee L"],
    "youtube_branded_products_categories": ["SUV", "vehicle"],
    "youtube_call_to_actions": [
        "visit",
        "learn more",
        "buy now",
        "order now",
        "search",
        "click",
        "subscribe",
    ],
}

# Optional: provide explicit hints for extractor, otherwise they are auto-generated from the video name
user_hints = ["SUV", "vehicle"]


try:
    clib_scribe_platform_conf = build_platform(clib_scribe_platform, **platform_params)

    if clib_scribe_platform_conf is not None:
        clib_scribe = build_clip_scribe(
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            clib_scribe_device=clib_scribe_device,
            clib_scribe_platform_name=clib_scribe_platform,
            user_hints=user_hints,
            clib_scribe_platform_conf=clib_scribe_platform_conf,
        )
        clib_scribe.run()

    else:
        raise NotImplementedError(f"no {clib_scribe_platform} available")

except Exception:
    traceback.print_exc()
    sys.exit(1)
