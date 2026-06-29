"""
entry point for local dev, just run with uv run main.py
"""


from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
from src.clip_scribe.build_clip_scribe_plalform import build_platform
from dotenv import load_dotenv, find_dotenv

import sys
import traceback

load_dotenv(find_dotenv())

video_name = "JEEP_-06ZPnShpv4 - 2024 Jeep Wrangler ｜ Jeep Adventure Days ｜ LEASE ｜ September.mp4"
video_path = f"input/{video_name}"
video_type = "car ad"

run_id = ""

clib_scribe_device = "mps"
clib_scribe_platform = "youtube"
clib_scribe_mode = "full"


platform_params = {
    "youtube_brand_name": "JEEP",
    "youtube_branded_products": ["Jeep Wrangler", "Jeep Adventure Days"],
    "youtube_branded_products_categories": [
        "Jeep Wrangler Pick Up Truck",
        "Jeep Wrangler Truck" "RAM HD Rebel Car",
        "Jeep Wrangler SUV",
        "Jeep Adventure Days Pick Up Truck",
        "Jeep Adventure Days Truck" "RAM HD Rebel Car",
        "Jeep Adventure Days SUV",
    ],
    "youtube_call_to_actions": [
        "learn more",
        "buy now",
        "order now",
        "purchase now",
        "get now",
        "apply now",
        "lease now",
        "accept now",
    ],
}

# Optional: provide explicit hints for extractor, otherwise they can be auto-generated from the video name
user_hints = [
    "suv",
    "vehicle",
    "car",
    "motor car",
    "motor vehicle",
    "truck",
    "license plate",
    "brand logo",
]
generate_hint_from_name = False


try:
    clib_scribe_platform_conf = build_platform(clib_scribe_platform, **platform_params)

    if clib_scribe_platform_conf is not None:
        builder = ClipScribeBuilder()

        clib_scribe = builder.build_clip_scribe(
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            clib_scribe_mode=clib_scribe_mode,
            clib_scribe_device=clib_scribe_device,
            clib_scribe_platform_name=clib_scribe_platform,
            user_hints=user_hints,
            generate_hint_from_name=generate_hint_from_name,
            clib_scribe_platform_conf=clib_scribe_platform_conf,
        )

        print(clib_scribe)
        clib_scribe.run(run_id=run_id)

    else:
        raise NotImplementedError(f"no {clib_scribe_platform} available")
except Exception:
    traceback.print_exc()
    sys.exit(1)
