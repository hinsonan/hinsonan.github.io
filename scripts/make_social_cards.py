"""Generate 1200x630 social preview cards for every post.

Writes assets/images/social/<slug>.png using each post's title and topic,
plus assets/images/social/default.png for pages without a card.
Re-run after adding or renaming a post.

Usage: python scripts/make_social_cards.py
Requires: Pillow
"""

import pathlib
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
POSTS = ROOT / "_posts"
OUT = ROOT / "assets" / "images" / "social"
LOGO = ROOT / "assets" / "images" / "robot.png"

WIDTH, HEIGHT = 1200, 630
PAD = 72
BG_TOP = (14, 17, 23)
BG_BOTTOM = (28, 33, 45)
ACCENT = (255, 94, 58)
TEXT = (240, 242, 245)
MUTED = (150, 158, 172)

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", str(pathlib.Path.home() / ".fonts")]
BOLD_NAMES = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
REGULAR_NAMES = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]


def find_font(names, size):
    for directory in FONT_DIRS:
        for name in names:
            for match in pathlib.Path(directory).rglob(name):
                return ImageFont.truetype(str(match), size)
    return ImageFont.load_default(size)


def front_matter(path):
    match = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.S)
    data = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
    return data


def wrap(draw, text, font, width):
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_title(draw, title, width, max_height):
    for size in range(76, 30, -4):
        font = find_font(BOLD_NAMES, size)
        lines = wrap(draw, title, font, width)
        if int(size * 1.18) * len(lines) <= max_height:
            return font, lines
    return font, lines


def background():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        t = y / HEIGHT
        color = tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
        draw.line([(0, y), (WIDTH, y)], fill=color)
    draw.rectangle([0, 0, 12, HEIGHT], fill=ACCENT)
    return image


def card(title, label, out_path):
    image = background()
    draw = ImageDraw.Draw(image)
    text_width = WIDTH - 2 * PAD - 40

    label_font = find_font(BOLD_NAMES, 28)
    draw.text((PAD, PAD), label.upper(), font=label_font, fill=ACCENT)

    logo_size = 64
    footer_y = HEIGHT - PAD - logo_size
    top = PAD + 60
    bottom = footer_y - 30

    title_font, lines = fit_title(draw, title, text_width, bottom - top)
    line_height = int(title_font.size * 1.18)
    y = top + (bottom - top - line_height * len(lines)) // 2
    for line in lines:
        draw.text((PAD, y), line, font=title_font, fill=TEXT)
        y += line_height

    footer_font = find_font(BOLD_NAMES, 30)
    sub_font = find_font(REGULAR_NAMES, 26)
    if LOGO.exists():
        logo = Image.open(LOGO).convert("RGBA").resize((logo_size, logo_size))
        mask = Image.new("L", (logo_size, logo_size), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, logo_size, logo_size], fill=255)
        image.paste(logo, (PAD, footer_y), mask)
    text_x = PAD + logo_size + 20
    draw.text((text_x, footer_y + 2), "Gradient Rage", font=footer_font, fill=TEXT)
    draw.text((text_x, footer_y + 36), "gradientrage.com", font=sub_font, fill=MUTED)

    image.save(out_path, optimize=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    card("ML, GPU Programming, and Software Engineering", "Gradient Rage", OUT / "default.png")
    for post in sorted(POSTS.glob("*")):
        if post.suffix not in (".md", ".markdown"):
            continue
        meta = front_matter(post)
        slug = re.sub(r"^\d{4}-\d{1,2}-\d{1,2}-", "", post.stem)
        card(meta.get("title", slug), meta.get("topic", "Gradient Rage"), OUT / f"{slug}.png")
        print(f"card: {slug}")


if __name__ == "__main__":
    main()
