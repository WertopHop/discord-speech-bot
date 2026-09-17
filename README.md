# discord-speech-bot
STT->LLM->TTS

Voice assistant for Discord: listens to a voice channel, transcribes speech (STT),
answers via LLM, speaks the reply (TTS). All models via **OpenRouter** (single API key).

```
speech -> VAD segmentation -> STT -> LLM (streamed)
-> sentence splitting -> TTS -> ffmpeg -> voice playback
```

- **Low latency**: streamed LLM tokens, per-sentence TTS (2-3 requests in flight),
  gapless PCM playback via ffmpeg (no temp files).
- **VAD** (webrtcvad) with pre-roll and configurable silence timeout.
- **Barge-in**: interrupt the bot by voice.
- **Per-channel conversation memory** (`!reset`).

STT / LLM / TTS models and the TTS voice are set in `.env`
(`STT_MODEL`, `LLM_MODEL`, `TTS_MODEL`, `TTS_VOICE`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env-example .env    # then fill it in
```

No system ffmpeg needed: `imageio-ffmpeg` ships the binary.

Fill `.env`:
- `DISCORD_TOKEN` — from the [Developer Portal](https://discord.com/developers/applications);
  also enable **Message Content Intent** there.
- `OPENROUTER_API_KEY` — from https://openrouter.ai/settings/keys
- `STT_MODEL`, `LLM_MODEL`, `TTS_MODEL`, `TTS_VOICE` — model slugs and voice from OpenRouter.
- Everything else (audio format, VAD timings, barge-in) is documented in `.env-example`.

Invite the bot with permissions: View Channels, Connect, Speak, Send Messages.
Self-deaf must stay **off** (bot must hear others).

## Run

```powershell
.venv\Scripts\python main.py
```

Commands: `!join`, `!leave`, `!stop`, `!reset`, `!say <text>` (TTS debug).

## Latency tuning (in `.env`)

- `SILENCE_MS` (700) — silence that ends an utterance (lower = faster replies).
- `MIN_SPEECH_MS` (200) — filters clicks.
- `BARGE_IN_MIN_MS` (350) — speech duration needed to interrupt the bot.
- `LLM_MAX_TOKENS` (512) — reply length cap.

## Limitations

- Voice receive uses Discord's legacy RTP path. Once Discord enforces E2EE
  (DAVE protocol) on guild voice channels, receiving will break for all
  libraries until DAVE decryption is implemented (pycord issue #3139).
- Utterances are processed sequentially; speech during a reply triggers
  barge-in and queues the new utterance.

## Structure

```
main.py                    # entry point
bot/
  config.py                # .env -> Settings
  bot.py                   # SpeechBot: commands, voice connect, capture
  pipeline.py              # STT -> LLM -> sentences -> TTS -> PCM -> playback
  openrouter.py            # OpenRouter client (STT / LLM stream / TTS stream)
  conversation.py          # dialogue history, prompt calibration stubs (TODO)
  audio/
    capture.py             # Sink: per-user VAD segmentation + barge-in
    vad.py                 # webrtcvad + energy fallback
    pcm.py                 # PCM/WAV conversion, ffmpeg decoder, AudioSource bridge
tests/                     # local-only checks (not committed)
```

Prompts: the default system prompt and calibration stubs live in
`bot/conversation.py`.


