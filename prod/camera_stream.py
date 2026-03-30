"""
MJPEG camera streaming over HTTP multipart/x-mixed-replace.

Works natively in all browsers including iPad Safari via <img src="/stream/left">.
No JavaScript decode needed. Cameras already output MJPG natively.
"""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable


async def mjpeg_stream(
    get_frame: Callable[[], bytes | None],
    fps: int = 15,
) -> AsyncGenerator[bytes]:
    """Async generator that yields MJPEG frames for multipart streaming.

    Args:
        get_frame: Callable returning latest JPEG bytes, or None if no frame.
        fps: Target frame rate for stream polling.
    """
    interval = 1.0 / fps
    while True:
        frame = get_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                b"\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(interval)
