"""Quick environment check: discord.py, voice receive, opus."""
import os
import sys

import discord

print("discord.py:", discord.__version__, "| python:", sys.version.split()[0])

from discord.voice_client import VoiceClient as V  # noqa: E402

print("start_recording:", hasattr(V, "start_recording"))
print("stop_recording:", hasattr(V, "stop_recording"))

import discord.sinks as sinks  # noqa: E402

print("sinks:", [x for x in dir(sinks) if "Sink" in x] or "EMPTY")

import discord.opus as opus  # noqa: E402

print("OpusDecoder:", hasattr(opus, "OpusDecoder"), "| OpusEncoder:", hasattr(opus, "OpusEncoder"))

bin_dir = os.path.join(os.path.dirname(discord.__file__), "bin")
print("bundled libopus:", os.listdir(bin_dir) if os.path.isdir(bin_dir) else "none")
