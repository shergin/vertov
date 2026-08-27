#!/usr/bin/env python3
"""Decode terminal pixel-protocol output back into PNG files — the harness
that lets automated QA *look at* vertov's pixel charts instead of trusting
the escape sequences.

Handles the three protocols malevich emits:
  kitty   APC `ESC _ G a=T,f=32,s=W,v=H,...;<base64 RGBA> ESC \\` (m=1 chains)
  iterm2  OSC 1337 `File=inline=1...:<base64 PNG> BEL`
  sixel   DCS `ESC P ... q ... ESC \\` (decoded via libsixel's sixel2png)

Usage: decode_pixels.py <capture-file> <outdir>
Writes img-000.png, img-001.png, ... and prints one line per image.
Raw-RGBA decoding needs Pillow; sixel needs `sixel2png` on PATH.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

ESC = b"\x1b"


def save_png(outdir: Path, count: int, data: bytes, kind: str) -> None:
    path = outdir / f"img-{count:03}.png"
    path.write_bytes(data)
    print(f"{path.name}: {kind}, {len(data)} bytes")


def rgba_to_png(rgba: bytes, width: int, height: int) -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.frombytes("RGBA", (width, height), rgba)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = Path(sys.argv[1]).read_bytes()
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0

    # --- kitty: chains of APC _G payloads until m=0 (or no m key).
    kitty_chunks = re.findall(rb"\x1b_G([^\x1b]*)\x1b\\", data)
    pending: bytes = b""
    meta: dict[str, int] = {}
    for chunk in kitty_chunks:
        options, _, payload = chunk.partition(b";")
        opts = dict(
            pair.split(b"=", 1) for pair in options.split(b",") if b"=" in pair
        )
        if b"s" in opts and b"v" in opts:
            meta = {"w": int(opts[b"s"]), "h": int(opts[b"v"])}
            pending = b""
        pending += payload
        if opts.get(b"m", b"0") != b"1" and meta:
            rgba = base64.b64decode(pending)
            if opts.get(b"o") == b"z" or rgba[:1] == b"\x78":
                import zlib

                rgba = zlib.decompress(rgba)
            png = rgba_to_png(rgba, meta["w"], meta["h"])
            save_png(outdir, count, png, f"kitty {meta['w']}x{meta['h']}")
            count += 1
            pending, meta = b"", {}

    # --- iTerm2: OSC 1337 File=...:<base64 PNG> terminated by BEL.
    for payload in re.findall(rb"\x1b\]1337;File=[^:]*:([A-Za-z0-9+/=]+)\x07", data):
        save_png(outdir, count, base64.b64decode(payload), "iterm2 png")
        count += 1

    # --- sixel: DCS ... q ... ST, decoded by libsixel.
    for blob in re.findall(rb"\x1bP[0-9;]*q[^\x1b]*\x1b\\", data):
        result = subprocess.run(
            ["sixel2png", "-i", "/dev/stdin", "-o", "/dev/stdout"],
            input=blob,
            capture_output=True,
            check=True,
        )
        save_png(outdir, count, result.stdout, "sixel")
        count += 1

    if count == 0:
        sys.exit("no pixel-protocol blocks found in the capture")


if __name__ == "__main__":
    main()
