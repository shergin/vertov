#!/usr/bin/env python3
"""Render an ANSI-colored terminal capture (e.g. `tmux capture-pane -e -p`)
to a PNG — screenshots of the TUI's cell chrome for eyes-on design review,
no GUI terminal required.

Handles the SGR subset ratatui emits: truecolor/256/16 fg+bg, bold, dim,
reversed, reset. Other CSI sequences are dropped.

Usage: ansi_screenshot.py <capture.txt> <out.png> [--bg 101010]
Requires Pillow; uses the system Menlo font on macOS.
"""

import re
import sys
from PIL import Image, ImageDraw, ImageFont

CSI = re.compile(r"\x1b\[([0-9;:]*)([A-Za-z])")

BASE16 = [
    (0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16),
    (36, 114, 200), (188, 63, 188), (17, 168, 205), (229, 229, 229),
    (102, 102, 102), (241, 76, 76), (35, 209, 139), (245, 245, 67),
    (59, 142, 234), (214, 112, 214), (41, 184, 219), (255, 255, 255),
]


def color256(index: int) -> tuple:
    if index < 16:
        return BASE16[index]
    if index < 232:
        index -= 16
        levels = [0, 95, 135, 175, 215, 255]
        return (levels[index // 36], levels[index // 6 % 6], levels[index % 6])
    gray = 8 + (index - 232) * 10
    return (gray, gray, gray)


def main() -> None:
    source, target = sys.argv[1], sys.argv[2]
    background = tuple(
        int(sys.argv[sys.argv.index("--bg") + 1][i : i + 2], 16)
        for i in (0, 2, 4)
    ) if "--bg" in sys.argv else (16, 16, 16)
    default_fg = (222, 220, 214)

    text = open(source, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")

    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14, index=1)
    cell_w = int(font.getlength("M"))
    cell_h = 18

    # First pass: strip sequences to find the grid size.
    columns = max((len(CSI.sub("", line)) for line in lines), default=80)
    image = Image.new("RGB", (columns * cell_w + 16, len(lines) * cell_h + 16), background)
    draw = ImageDraw.Draw(image)

    for row, line in enumerate(lines):
        x = 0
        fg, bg, is_bold, reversed_ = default_fg, None, False, False
        position = 0
        for match in CSI.finditer(line):
            chunk = line[position : match.start()]
            position = match.end()
            for ch in chunk:
                cx, cy = 8 + x * cell_w, 8 + row * cell_h
                actual_fg, actual_bg = (bg or background, fg) if reversed_ else (fg, bg)
                if actual_bg:
                    draw.rectangle([cx, cy, cx + cell_w, cy + cell_h], fill=actual_bg)
                draw.text((cx, cy), ch, fill=actual_fg, font=bold if is_bold else font)
                x += 1
            if match.group(2) != "m":
                continue
            params = [p for p in match.group(1).split(";")] or [""]
            i = 0
            while i < len(params):
                p = params[i] or "0"
                if p == "0":
                    fg, bg, is_bold, reversed_ = default_fg, None, False, False
                elif p == "1":
                    is_bold = True
                elif p == "7":
                    reversed_ = True
                elif p == "27":
                    reversed_ = False
                elif p in ("22", "2"):
                    is_bold = False
                elif p == "39":
                    fg = default_fg
                elif p == "49":
                    bg = None
                elif p == "38" and params[i + 1 : i + 2] == ["2"]:
                    fg = tuple(int(v) for v in params[i + 2 : i + 5])
                    i += 4
                elif p == "48" and params[i + 1 : i + 2] == ["2"]:
                    bg = tuple(int(v) for v in params[i + 2 : i + 5])
                    i += 4
                elif p == "38" and params[i + 1 : i + 2] == ["5"]:
                    fg = color256(int(params[i + 2]))
                    i += 2
                elif p == "48" and params[i + 1 : i + 2] == ["5"]:
                    bg = color256(int(params[i + 2]))
                    i += 2
                elif p.isdigit() and 30 <= int(p) <= 37:
                    fg = BASE16[int(p) - 30]
                elif p.isdigit() and 90 <= int(p) <= 97:
                    fg = BASE16[int(p) - 82]
                elif p.isdigit() and 40 <= int(p) <= 47:
                    bg = BASE16[int(p) - 40]
                i += 1
        for ch in line[position:]:
            cx, cy = 8 + x * cell_w, 8 + row * cell_h
            actual_fg, actual_bg = (bg or background, fg) if reversed_ else (fg, bg)
            if actual_bg:
                draw.rectangle([cx, cy, cx + cell_w, cy + cell_h], fill=actual_bg)
            draw.text((cx, cy), ch, fill=actual_fg, font=bold if is_bold else font)
            x += 1

    image.save(target)
    print(f"{target}: {columns}x{len(lines)} cells")


if __name__ == "__main__":
    main()
