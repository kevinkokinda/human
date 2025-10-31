import argparse
import sys
import textwrap
from pathlib import Path

from .corpus import CorpusProfile
from .engine import Humanizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="humanizer", description="Convert formal AI-like writing into casual human phrasing.")
    parser.add_argument("text", nargs="*", help="Text to humanize. You can also pipe or use --file.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file.")
    parser.add_argument("-c", "--creativity", type=float, default=0.65, help="Creativity level between 0 and 1.")
    parser.add_argument("-s", "--seed", type=int, help="Deterministic seed.")
    parser.add_argument("-w", "--wrap", type=int, default=0, help="Optional line wrap width.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch the terminal UI for live humanizing.")
    parser.add_argument("--corpus", type=Path, action="append", help="Optional path(s) to human-written text corpora that guide phrasing.")
    return parser


def format_output(text: str, width: int) -> str:
    if width and width > 10:
        wrapper = textwrap.TextWrapper(width=width)
        return "\n".join(wrapper.fill(line) if line.strip() else "" for line in text.split("\n"))
    return text


def read_from_source(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.text:
        return " ".join(args.text)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _basic_interactive_loop(humanizer: Humanizer, wrap: int) -> None:
    print("Interactive humanizer (basic mode)")
    print("Type your text. Use ::go on a new line to process, ::quit to exit.")
    while True:
        print("-" * 40)
        captured: list[str] = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                return
            command = line.strip().lower()
            if command == "::quit":
                return
            if command == "::go":
                break
            captured.append(line)
        raw = "\n".join(captured).strip()
        if not raw:
            print("Nothing captured.")
            continue
        output = humanizer.humanize(raw)
        print("\nHumanized output\n")
        print(format_output(output, wrap))
        print()


def interactive_loop(humanizer: Humanizer, wrap: int) -> None:
    try:
        _tui_loop(humanizer, wrap)
    except (ImportError, RuntimeError) as exc:
        print(f"[fallback] {exc}")
        _basic_interactive_loop(humanizer, wrap)


def _tui_loop(humanizer: Humanizer, wrap: int) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("TUI requires an interactive terminal; falling back to basic mode.")

    try:
        import curses
    except ImportError as import_exc:  # pragma: no cover - platform dependent
        raise ImportError("Curses is not available on this platform; falling back to basic mode.") from import_exc

    def _render(stdscr: "curses.window", buffer: str, output: str) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        title = "Humanizer TUI — Ctrl+Q quit • Ctrl+L clear • Output updates live"
        stdscr.addnstr(0, 0, title.ljust(width), width)

        input_label_line = 1
        stdscr.addnstr(input_label_line, 0, "Input".ljust(width), width, curses.A_BOLD)

        input_start = input_label_line + 1
        margin = 1
        usable_height = max(height - (input_start + 2), 4)
        input_height = max(usable_height // 2, 3)
        output_start = input_start + input_height + 1
        output_height = height - output_start - 1

        stdscr.hline(output_start - 1, 0, curses.ACS_HLINE, width)
        stdscr.addnstr(output_start, 0, "Humanized".ljust(width), width, curses.A_BOLD)
        output_body_start = output_start + 1

        input_lines = buffer.split("\n") or [""]
        for row in range(input_height):
            line = input_lines[row] if row < len(input_lines) else ""
            stdscr.addnstr(input_start + row, margin, line[: width - margin * 2], width - margin * 2)

        display_width = wrap if wrap and wrap > 0 else width - margin * 2
        formatted_output = format_output(output, display_width)
        output_lines = formatted_output.split("\n") if formatted_output else []
        for row in range(output_height):
            line = output_lines[row] if row < len(output_lines) else ""
            stdscr.addnstr(output_body_start + row, margin, line[: width - margin * 2], width - margin * 2)

        last_line = input_lines[-1] if input_lines else ""
        cursor_y = input_start + min(len(input_lines) - 1, input_height - 1)
        cursor_x = margin + min(len(last_line), width - margin * 2 - 1)
        stdscr.move(cursor_y, cursor_x)
        stdscr.refresh()

    def _loop(stdscr: "curses.window") -> None:
        curses.curs_set(1)
        stdscr.keypad(True)
        stdscr.timeout(-1)
        buffer: list[str] = []
        cached_text = ""
        cached_output = ""

        while True:
            current_text = "".join(buffer)
            if current_text != cached_text:
                cached_text = current_text
                cached_output = humanizer.humanize(current_text) if current_text.strip() else ""
            _render(stdscr, current_text, cached_output)
            ch = stdscr.get_wch()

            if ch == curses.KEY_RESIZE:
                continue

            if isinstance(ch, str):
                if ch == "\x11":  # Ctrl+Q
                    return
                if ch in ("\b", "\x7f"):  # Backspace variants
                    if buffer:
                        buffer.pop()
                    continue
                if ch == "\x0c":  # Ctrl+L
                    buffer.clear()
                    continue
                buffer.append(ch)
                continue

            if ch in (curses.KEY_BACKSPACE, curses.KEY_DC):
                if buffer:
                    buffer.pop()
                continue

    curses.wrapper(_loop)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    corpus_profile: CorpusProfile | None = None
    if args.corpus:
        corpus_segments: list[str] = []
        for corpus_path in args.corpus:
            try:
                corpus_segments.append(corpus_path.read_text(encoding="utf-8"))
            except OSError as exc:
                parser.error(f"Failed to read corpus {corpus_path}: {exc}")
        combined = "\n\n".join(corpus_segments)
        if combined.strip():
            corpus_profile = CorpusProfile(combined)
    humanizer = Humanizer(creativity=args.creativity, seed=args.seed, corpus=corpus_profile)
    if args.interactive:
        interactive_loop(humanizer, args.wrap)
        return 0
    source = read_from_source(args)
    if not source.strip():
        parser.print_help()
        return 1
    result = humanizer.humanize(source)
    print(format_output(result, args.wrap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
