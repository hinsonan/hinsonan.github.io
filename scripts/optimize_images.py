"""Make web-sized WebP copies of large post images.

For every PNG over SIZE_LIMIT bytes, writes a WebP next to it, scaled down
so the longest side is at most MAX_SIDE. The original PNG is kept so posts
can link to the full-resolution file.

Usage: python scripts/optimize_images.py [files...]
Requires: Pillow with WebP support
"""

import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / "assets" / "images"
SIZE_LIMIT = 300 * 1024
MAX_SIDE = 2400
QUALITY = 88


def convert(path):
    out = path.with_suffix(".webp")
    image = Image.open(path)
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    image.save(out, "WEBP", quality=QUALITY, method=6)
    before, after = path.stat().st_size, out.stat().st_size
    print(f"{path.relative_to(ROOT)}: {before // 1024} KB -> {after // 1024} KB {image.size}")


def main(args):
    if args:
        targets = [pathlib.Path(a).resolve() for a in args]
    else:
        targets = [
            p for p in IMAGES.rglob("*.png")
            if p.stat().st_size > SIZE_LIMIT and "social" not in p.parts
        ]
    for path in sorted(targets):
        convert(path)


if __name__ == "__main__":
    main(sys.argv[1:])
