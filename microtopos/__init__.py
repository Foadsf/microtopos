"""MicroTopos: encoderless computational microscopy and surface metrology."""

__version__ = "0.8.0"


def safe_console() -> None:
    """Never crash on printing: characters the console cannot encode (e.g. Greek letters on a Windows cp1252
    console) are replaced instead of raising UnicodeEncodeError."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a text stream, or detached (windowed program)
            pass
