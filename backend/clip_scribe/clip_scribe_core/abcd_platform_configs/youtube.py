from typing import List
from .base import BasePlatformConf


class YouTubePlatformConf(BasePlatformConf):
    brand_name: str
    branded_products: List[str]
    branded_products_categories: List[str]
    call_to_actions: List[str]
