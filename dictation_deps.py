"""Optional imports shared by the dictation client modules."""

__all__ = ("aiohttp", "ecodes", "evdev")

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import evdev
    from evdev import ecodes
except ImportError:
    evdev = None
    ecodes = None
