"""Generates the three placeholder camera frames used by the demo and the tests.

    python scenarios/make_images.py

Needs Pillow. The PNGs are committed, so this only has to run if you want to regenerate them.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "images"
W, H = 640, 480
BG, PANEL, DARK, LIGHT = (38, 42, 48), (92, 98, 106), (24, 26, 30), (210, 214, 220)


def base(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((40, 40, W - 40, H - 60), fill=PANEL, outline=LIGHT, width=3)
    d.text((52, H - 46), title, fill=LIGHT)
    return img, d


def panel_disconnected_cable() -> Image.Image:
    img, d = base("CONTROL PANEL  MODEL CP-200")
    d.rectangle((80, 90, 330, 250), fill=DARK, outline=LIGHT, width=2)
    d.text((100, 105), "POWER IN", fill=LIGHT)
    d.rectangle((150, 150, 260, 210), fill=(10, 10, 12), outline=LIGHT, width=2)  # empty socket
    d.ellipse((380, 100, 420, 140), fill=(20, 45, 20), outline=LIGHT, width=2)  # status LED: off
    d.text((372, 148), "STATUS", fill=LIGHT)
    pts = [(300, 300), (360, 330), (330, 370), (250, 350), (230, 300), (250, 260)]
    d.line(pts, fill=(230, 150, 40), width=9, joint="curve")  # cable dangling loose
    d.rectangle((228, 246, 272, 276), fill=LIGHT, outline=DARK, width=2)  # unplugged connector
    return img


def machine_overheating_fan() -> Image.Image:
    img, d = base("MOTOR UNIT  M-7")
    cx, cy, r = 240, 240, 110
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=DARK, outline=LIGHT, width=4)
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):  # four stopped blades
        d.polygon([(cx, cy), (cx + dx * 95 - dy * 28, cy + dy * 95 + dx * 28),
                   (cx + dx * 95 + dy * 28, cy + dy * 95 - dx * 28)], fill=(120, 126, 134))
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=LIGHT)
    for i in range(0, 220, 14):  # dust-clogged vent slats
        d.rectangle((420, 80 + i, 590, 88 + i), fill=(110, 92, 70))
    d.ellipse((470, 350, 520, 400), fill=(220, 30, 30), outline=LIGHT, width=3)  # warning lamp
    d.text((452, 408), "TEMP HIGH", fill=(255, 120, 120))
    return img


def breaker_tripped() -> Image.Image:
    img, d = base("ELECTRICAL PANEL  DB-3")
    for i in range(6):
        x = 90 + i * 85
        d.rectangle((x, 110, x + 55, 330), fill=DARK, outline=LIGHT, width=2)
        if i == 3:  # tripped: handle sits mid-way, red marker
            d.rectangle((x + 8, 200, x + 47, 240), fill=(200, 60, 60), outline=LIGHT, width=2)
            d.text((x + 4, 340), "TRIPPED", fill=(255, 120, 120))
        else:  # on
            d.rectangle((x + 8, 125, x + 47, 165), fill=(70, 170, 90), outline=LIGHT, width=2)
        d.text((x + 14, 92), f"C{i + 1}", fill=LIGHT)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in (("panel_disconnected_cable.png", panel_disconnected_cable),
                     ("machine_overheating_fan.png", machine_overheating_fan),
                     ("breaker_tripped.png", breaker_tripped)):
        fn().save(OUT / name)
        print("wrote", OUT / name)
