"""Entry point of discord-speech-bot."""
import logging
import sys

from bot.bot import SpeechBot
from bot.config import ConfigError, get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        sys.exit(1)

    bot = SpeechBot(settings)
    # log_handler=None: keep our logging config
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
