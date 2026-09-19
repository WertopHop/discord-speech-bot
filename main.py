"""Entry point of discord-speech-bot."""
import logging
import sys
from pathlib import Path

from bot.bot import SpeechBot
from bot.config import ConfigError, get_settings


class _ConsoleFilter(logging.Filter):
    """Console shows only our own warnings/errors + library errors.

    Everything else (verbose diagnostics) goes to diag.log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("bot.") or record.levelno >= logging.ERROR


def _setup_logging() -> None:
    diag_path = Path(__file__).resolve().parent / "diag.log"
    file_handler = logging.FileHandler(diag_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.addFilter(_ConsoleFilter())
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.getLogger(__name__).info("diagnostics file: %s", diag_path)


def main() -> None:
    _setup_logging()
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        sys.exit(1)

    bot = SpeechBot(settings)
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
