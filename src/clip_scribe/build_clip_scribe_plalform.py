from src.clip_scribe.platform_configs import *  # noqa ignore


def build_platform(platform_name: str, **kwargs):
    platform = None

    if platform_name == "youtube":
        youtube_brand_name = kwargs.get("youtube_brand_name")
        youtube_branded_products = kwargs.get("youtube_branded_products", [])
        youtube_branded_products_categories = kwargs.get(
            "youtube_branded_products_categories", []
        )
        youtube_call_to_actions = kwargs.get("youtube_call_to_actions", [])

        YouTubePlatformConf(  # noqa ignore
            brand_name=youtube_brand_name,
            branded_products=youtube_branded_products,
            branded_products_categories=youtube_branded_products_categories,
            call_to_actions=youtube_call_to_actions,
        )

    return platform
