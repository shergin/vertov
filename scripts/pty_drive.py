#!/usr/bin/env python3
"""Drive a TUI in a real, properly sized pty and record every byte it emits
— unlike `script`(1), which inherits a 0x0 window when the caller has no
tty, and unlike tmux, which filters image escape sequences out of its panes.
The recording feeds `decode_pixels.py` for eyes-on chart verification.

Usage:
  pty_drive.py --size 110x28 --out capture.raw --keys "2@2.0,q@4.0" -- cmd args...

Keys are `text@seconds` pairs (seconds from start); `Esc`, `Enter`, `Space`,
and `Tab` are understood as names.
"""

import argparse
import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

NAMES = {"Esc": "\x1b", "Enter": "\r", "Space": " ", "Tab": "\t"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="110x28")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keys", default="")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    columns, rows = (int(part) for part in args.size.split("x"))

    schedule = []
    if args.keys:
        for entry in args.keys.split(","):
            text, _, at = entry.partition("@")
            schedule.append((float(at or 0), NAMES.get(text, text)))
    schedule.sort()

    pid, master = pty.fork()
    if pid == 0:
        os.execvp(command[0], command)
    winsize = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(master, termios.TIOCSWINSZ, winsize)

    start = time.monotonic()
    sent = 0
    with open(args.out, "wb") as out:
        while True:
            now = time.monotonic() - start
            while sent < len(schedule) and schedule[sent][0] <= now:
                os.write(master, schedule[sent][1].encode())
                sent += 1
            if now > args.timeout:
                break
            timeout = 0.05
            if sent < len(schedule):
                timeout = min(timeout, max(0.0, schedule[sent][0] - now))
            ready, _, _ = select.select([master], [], [], timeout)
            if master in ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out.write(chunk)
    finished, status = os.waitpid(pid, os.WNOHANG)
    if finished == 0:
        os.kill(pid, 15)
        os.waitpid(pid, 0)
        status = 0
    sys.exit(os.waitstatus_to_exitcode(status) if finished else 0)


if __name__ == "__main__":
    main()
