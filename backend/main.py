"""
entry point for local dev, just run with uv run main.py
"""


from clip_scribe.clip_scribe_core.build_clip_scribe import ClipScribeBuilder
from clip_scribe.clip_scribe_core.build_clip_scribe_plalform import build_platform
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import sys
import traceback

load_dotenv(find_dotenv())

# local dev, make absolute path to local sa at the root
PROJECT_ROOT = Path(__file__).resolve().parents[0]
REPO_ROOT = PROJECT_ROOT.parent
raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if raw and not Path(raw).is_absolute():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((REPO_ROOT / raw).resolve())

video_name = "01KXMKH0X9AH0D6H5E0JCG1ZKV.mp4"
video_path = f"input/{video_name}"
video_type = "car commercial"

run_id = ""

clib_scribe_device = "mps"
clib_scribe_platform = "youtube"
clib_scribe_mode = "full"
extraction_mode = "no_objects"
parsing_mode = "exploratory"


platform_params = {
    "youtube_brand_name": "Jeep",
    "youtube_branded_products": ["Jeep Grand Cherokee"],
    "youtube_branded_products_categories": [
        "Jeep Grand Cherokee Pick Up Truck",
        "Jeep Grand Cherokee Truck",
        "Jeep Grand Cherokee Car",
        "Jeep Grand Cherokee SUV",
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
        builder = ClipScribeBuilder(device=clib_scribe_device)

        clib_scribe = builder.build_clip_scribe(
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            clib_scribe_mode=clib_scribe_mode,
            extraction_mode=extraction_mode,
            parsing_mode=parsing_mode,
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
