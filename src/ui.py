"""Background image and page styling for the web app.

Put your own photo in the project folder as  background.jpg  (or .png / .webp).
Keep it under about 1 MB so the app stays fast. Without a photo, a built-in
sunset-houses picture is used.
"""
from __future__ import annotations
import base64
import mimetypes
from functools import lru_cache
from urllib.parse import quote

from src import config

DEFAULT_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMid slice'>
<defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'>
<stop offset='0' stop-color='#1b2f5b'/><stop offset='0.55' stop-color='#e0735a'/><stop offset='1' stop-color='#f6c177'/></linearGradient></defs>
<rect width='1600' height='900' fill='url(#sky)'/>
<circle cx='1180' cy='560' r='95' fill='#fde7b0' opacity='0.9'/>
<path d='M0 640 Q300 560 600 630 T1200 620 T1600 600 V900 H0 Z' fill='#2c3e5c'/>
<g fill='#16223b'>
<rect x='90' y='560' width='230' height='200'/><polygon points='70,562 205,450 340,562'/>
<rect x='430' y='600' width='170' height='160'/><polygon points='415,602 515,520 615,602'/>
<rect x='700' y='520' width='280' height='240'/><polygon points='680,522 840,390 1000,522'/>
<rect x='1090' y='590' width='200' height='170'/><polygon points='1072,592 1190,500 1308,592'/>
<rect x='1370' y='570' width='190' height='190'/><polygon points='1352,572 1465,470 1578,572'/>
</g>
<g fill='#ffd56b'>
<rect x='130' y='610' width='36' height='46'/><rect x='244' y='610' width='36' height='46'/>
<rect x='470' y='640' width='30' height='38'/><rect x='530' y='640' width='30' height='38'/>
<rect x='745' y='575' width='44' height='56'/><rect x='890' y='575' width='44' height='56'/><rect x='815' y='660' width='50' height='100'/>
<rect x='1130' y='630' width='34' height='44'/><rect x='1215' y='630' width='34' height='44'/>
<rect x='1410' y='615' width='36' height='46'/><rect x='1490' y='615' width='36' height='46'/>
</g>
<rect y='760' width='1600' height='140' fill='#0f1a2e'/>
</svg>"""


@lru_cache(maxsize=1)
def _background_url() -> str:
    for ext in ("jpg", "jpeg", "png", "webp"):
        f = config.ROOT / f"background.{ext}"
        if f.exists():
            mime = mimetypes.guess_type(f.name)[0] or "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()
    return "data:image/svg+xml;utf8," + quote(DEFAULT_SVG)


def page_css() -> str:
    return f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(8, 16, 36, 0.35), rgba(8, 16, 36, 0.35)), url("{_background_url()}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
    background: rgba(255, 255, 255, 0.94);
    border-radius: 20px;
    padding: 2rem 2.6rem 2.4rem 2.6rem !important;
    margin-top: 1.4rem;
    margin-bottom: 1.4rem;
    max-width: 1150px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
}}
</style>
"""
