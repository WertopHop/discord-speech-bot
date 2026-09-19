"""Console markers: the only things printed to the terminal.

print() without timestamps - all verbose diagnostics go to diag.log instead.
"""


def mark(message: str) -> None:
    print(f"*** {message}", flush=True)


def heard(text: str) -> None:
    """What STT produced (this exact text goes to the LLM)."""
    print(f">>> heard: {text}", flush=True)


def say(text: str) -> None:
    """The sentence being synthesized for playback right now."""
    print(f"<<< say: {text}", flush=True)