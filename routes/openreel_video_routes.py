"""
OpenReel Video API Routes

Registers API endpoints for serving OpenReel videos and the OpenReel app.
"""

import os
import mimetypes
import logging
from aiohttp import web
from server import PromptServer
import folder_paths

logger = logging.getLogger("WAS.OpenReel.Routes")

# Track temp video sessions: session_id -> {"video_path": str, "audio_path": str|None}
_openreel_sessions = {}


def _get_openreel_app_dir():
    """Get the path to the built OpenReel app static files."""
    return os.path.join(os.path.dirname(__file__), "..", "apps", "openreel_app")


def _validate_session_id(session_id: str) -> bool:
    """Validate session_id format to prevent path traversal."""
    if not session_id or len(session_id) > 32:
        return False
    return all(c.isalnum() or c in ("-", "_") for c in session_id)


@PromptServer.instance.routes.get("/was/openreel_video/serve")
async def serve_openreel_video(request):
    """Serve a temp video file with HTTP Range support for seeking."""
    session_id = request.query.get("session_id", "")

    if not _validate_session_id(session_id):
        return web.json_response({"error": "Invalid session_id"}, status=400)

    session = _openreel_sessions.get(session_id)
    if not session or not session.get("video_path"):
        return web.json_response({"error": "No video for session"}, status=404)

    video_path = session["video_path"]

    # Verify the file is within the temp directory (prevent path traversal)
    temp_dir = os.path.realpath(folder_paths.get_temp_directory())
    real_path = os.path.realpath(video_path)
    if not real_path.startswith(temp_dir):
        return web.json_response({"error": "Access denied"}, status=403)

    if not os.path.exists(real_path):
        return web.json_response({"error": "Video file not found"}, status=404)

    file_size = os.path.getsize(real_path)
    content_type = "video/mp4"

    # Handle HTTP Range requests for video seeking
    range_header = request.headers.get("Range")
    if range_header:
        try:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(real_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            response = web.Response(
                body=data,
                status=206,
                content_type=content_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )
            return response
        except (ValueError, IndexError):
            pass

    # Full file response
    with open(real_path, "rb") as f:
        data = f.read()

    return web.Response(
        body=data,
        content_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


@PromptServer.instance.routes.get("/was/openreel_video/serve_audio")
async def serve_openreel_audio(request):
    """Serve a temp audio file for OpenReel."""
    session_id = request.query.get("session_id", "")

    if not _validate_session_id(session_id):
        return web.json_response({"error": "Invalid session_id"}, status=400)

    session = _openreel_sessions.get(session_id)
    if not session or not session.get("audio_path"):
        return web.json_response({"error": "No audio for session"}, status=404)

    audio_path = session["audio_path"]

    # Verify the file is within the temp directory
    temp_dir = os.path.realpath(folder_paths.get_temp_directory())
    real_path = os.path.realpath(audio_path)
    if not real_path.startswith(temp_dir):
        return web.json_response({"error": "Access denied"}, status=403)

    if not os.path.exists(real_path):
        return web.json_response({"error": "Audio file not found"}, status=404)

    with open(real_path, "rb") as f:
        data = f.read()

    return web.Response(
        body=data,
        content_type="audio/wav",
        headers={
            "Content-Length": str(len(data)),
        },
    )


@PromptServer.instance.routes.get("/was/openreel_video/app/{path:.*}")
async def serve_openreel_app(request):
    """Serve the built OpenReel app static files."""
    path = request.match_info.get("path", "index.html")
    if not path:
        path = "index.html"

    app_dir = _get_openreel_app_dir()
    file_path = os.path.join(app_dir, path)

    # Prevent path traversal
    real_app_dir = os.path.realpath(app_dir)
    real_file_path = os.path.realpath(file_path)
    if not real_file_path.startswith(real_app_dir):
        return web.json_response({"error": "Access denied"}, status=403)

    if not os.path.exists(real_file_path) or not os.path.isfile(real_file_path):
        return web.json_response({"error": "File not found"}, status=404)

    content_type, _ = mimetypes.guess_type(real_file_path)
    if content_type is None:
        content_type = "application/octet-stream"

    # Ensure correct MIME types for web assets
    if real_file_path.endswith(".js"):
        content_type = "application/javascript"
    elif real_file_path.endswith(".mjs"):
        content_type = "application/javascript"
    elif real_file_path.endswith(".css"):
        content_type = "text/css"
    elif real_file_path.endswith(".wasm"):
        content_type = "application/wasm"
    elif real_file_path.endswith(".svg"):
        content_type = "image/svg+xml"
    elif real_file_path.endswith(".json"):
        content_type = "application/json"

    with open(real_file_path, "rb") as f:
        data = f.read()

    headers = {
        "Content-Length": str(len(data)),
        "Cache-Control": "public, max-age=3600",
    }

    return web.Response(
        body=data,
        content_type=content_type,
        headers=headers,
    )


# Export sessions dict so nodes can access it
def get_sessions():
    """Get the sessions dictionary for use by nodes."""
    return _openreel_sessions


logger.info("[OpenReel Video] API routes registered")
