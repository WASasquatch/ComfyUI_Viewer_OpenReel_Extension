"""
OpenReel Video API Routes

Registers API endpoints for serving OpenReel videos and the OpenReel app.
"""

import os
import json
import uuid
import shutil
import mimetypes
import logging
import subprocess
import tempfile
import asyncio
import time
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


@PromptServer.instance.routes.get("/was/openreel_video/serve_image")
async def serve_openreel_image(request):
    """Serve an individual image file from a session directory."""
    session_id = request.query.get("session_id", "")
    filename = request.query.get("filename", "")

    if not _validate_session_id(session_id):
        return web.json_response({"error": "Invalid session_id"}, status=400)

    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return web.json_response({"error": "Invalid filename"}, status=400)

    session = _openreel_sessions.get(session_id)
    if not session:
        return web.json_response({"error": "No session found"}, status=404)

    image_dir = session.get("image_dir")
    if not image_dir:
        return web.json_response({"error": "No images for session"}, status=404)

    image_path = os.path.join(image_dir, filename)

    # Verify the file is within the temp directory (prevent path traversal)
    temp_dir = os.path.realpath(folder_paths.get_temp_directory())
    real_path = os.path.realpath(image_path)
    if not real_path.startswith(temp_dir):
        return web.json_response({"error": "Access denied"}, status=403)

    if not os.path.exists(real_path):
        return web.json_response({"error": "Image file not found"}, status=404)

    content_type, _ = mimetypes.guess_type(real_path)
    if content_type is None:
        content_type = "image/png"

    with open(real_path, "rb") as f:
        data = f.read()

    return web.Response(
        body=data,
        content_type=content_type,
        headers={
            "Content-Length": str(len(data)),
            "Cache-Control": "public, max-age=3600",
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


# ---------------------------------------------------------------------------
# Backend FFmpeg Render Engine endpoints
# ---------------------------------------------------------------------------

# Track active render sessions: render_id -> { dir, frames, audio, settings, ... }
_render_sessions = {}


def _find_ffmpeg():
    """Find the system FFmpeg binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Common locations
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_ffprobe():
    """Find the system ffprobe binary."""
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in ["/usr/bin/ffprobe", "/usr/local/bin/ffprobe"]:
        if os.path.isfile(candidate):
            return candidate
    return None


@PromptServer.instance.routes.get("/was/openreel_video/ffmpeg_check")
async def ffmpeg_check(request):
    """Check if native FFmpeg is available and return its capabilities."""
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        return web.json_response({
            "available": False,
            "error": "FFmpeg not found on system PATH",
        })

    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True, text=True, timeout=5,
        )
        version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"

        # Check for hardware acceleration
        hw_result = subprocess.run(
            [ffmpeg_path, "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        hwaccels = [
            line.strip()
            for line in (hw_result.stdout or "").split("\n")[1:]
            if line.strip()
        ]

        return web.json_response({
            "available": True,
            "path": ffmpeg_path,
            "version": version_line,
            "hwaccels": hwaccels,
            "ffprobe": _find_ffprobe() is not None,
        })
    except Exception as e:
        return web.json_response({
            "available": False,
            "error": str(e),
        })


@PromptServer.instance.routes.post("/was/openreel_video/render_init")
async def render_init(request):
    """
    Initialize a backend render session.

    Expects JSON body:
    {
      "width": 1920, "height": 1080, "frameRate": 30,
      "totalFrames": 300,
      "format": "mp4",
      "videoBitrate": "12000k",
      "audioBitrate": "192k"
    }
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    render_id = uuid.uuid4().hex[:16]
    render_dir = tempfile.mkdtemp(prefix="was_openreel_render_")

    frames_dir = os.path.join(render_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    _render_sessions[render_id] = {
        "dir": render_dir,
        "frames_dir": frames_dir,
        "frame_count": 0,
        "total_frames": data.get("totalFrames", 0),
        "audio_path": None,
        "settings": {
            "width": data.get("width", 1920),
            "height": data.get("height", 1080),
            "frameRate": data.get("frameRate", 30),
            "format": data.get("format", "mp4"),
            "videoBitrate": data.get("videoBitrate", "12000k"),
            "audioBitrate": data.get("audioBitrate", "192k"),
        },
        "status": "receiving",
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "created_at": time.time(),
    }

    logger.info(f"[OpenReel Render] Session {render_id} initialized: {data.get('width')}x{data.get('height')} @ {data.get('frameRate')}fps, {data.get('totalFrames')} frames")

    return web.json_response({
        "render_id": render_id,
        "status": "receiving",
    })


@PromptServer.instance.routes.post("/was/openreel_video/render_frame")
async def render_frame(request):
    """
    Receive a single rendered frame (JPEG binary) via multipart upload.

    Form fields:
      - render_id: str
      - frame_index: int
      - frame: file (JPEG binary)
    """
    try:
        reader = await request.multipart()
    except Exception:
        return web.json_response({"error": "Expected multipart data"}, status=400)

    render_id = None
    frame_index = None
    frame_data = None

    async for part in reader:
        if part.name == "render_id":
            render_id = (await part.text()).strip()
        elif part.name == "frame_index":
            frame_index = int((await part.text()).strip())
        elif part.name == "frame":
            frame_data = await part.read(decode=False)

    if not render_id or frame_index is None or not frame_data:
        return web.json_response({"error": "Missing render_id, frame_index, or frame data"}, status=400)

    session = _render_sessions.get(render_id)
    if not session:
        return web.json_response({"error": "Unknown render session"}, status=404)

    # Write frame to disk
    frame_filename = f"frame_{frame_index:06d}.jpg"
    frame_path = os.path.join(session["frames_dir"], frame_filename)
    with open(frame_path, "wb") as f:
        f.write(frame_data)

    session["frame_count"] += 1
    total = session["total_frames"]
    if total > 0:
        session["progress"] = session["frame_count"] / total * 0.7

    return web.json_response({
        "status": "ok",
        "frames_received": session["frame_count"],
    })


@PromptServer.instance.routes.post("/was/openreel_video/render_audio")
async def render_audio(request):
    """
    Receive audio (WAV binary) for the render session.

    Form fields:
      - render_id: str
      - audio: file (WAV binary)
    """
    try:
        reader = await request.multipart()
    except Exception:
        return web.json_response({"error": "Expected multipart data"}, status=400)

    render_id = None
    audio_data = None

    async for part in reader:
        if part.name == "render_id":
            render_id = (await part.text()).strip()
        elif part.name == "audio":
            audio_data = await part.read(decode=False)

    if not render_id or not audio_data:
        return web.json_response({"error": "Missing render_id or audio data"}, status=400)

    session = _render_sessions.get(render_id)
    if not session:
        return web.json_response({"error": "Unknown render session"}, status=404)

    audio_path = os.path.join(session["dir"], "audio.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_data)

    session["audio_path"] = audio_path
    logger.info(f"[OpenReel Render] Session {render_id}: received audio ({len(audio_data)} bytes)")

    return web.json_response({"status": "ok"})


@PromptServer.instance.routes.post("/was/openreel_video/render_finalize")
async def render_finalize(request):
    """
    Finalize the render: encode all frames + optional audio into a video file
    using native FFmpeg. Returns the output filename once complete.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    render_id = data.get("render_id")
    if not render_id:
        return web.json_response({"error": "Missing render_id"}, status=400)

    session = _render_sessions.get(render_id)
    if not session:
        return web.json_response({"error": "Unknown render session"}, status=404)

    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        session["status"] = "error"
        session["error"] = "FFmpeg not found"
        return web.json_response({"error": "FFmpeg not found"}, status=500)

    settings = session["settings"]
    fmt = settings.get("format", "mp4")
    output_filename = f"openreel_render_{render_id}.{fmt}"
    output_path = os.path.join(session["dir"], output_filename)

    session["status"] = "encoding"
    session["progress"] = 0.7

    frames_pattern = os.path.join(session["frames_dir"], "frame_%06d.jpg")

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(settings["frameRate"]),
        "-i", frames_pattern,
    ]

    if session["audio_path"] and os.path.exists(session["audio_path"]):
        cmd.extend(["-i", session["audio_path"]])

    cmd.extend(["-threads", "0"])

    if fmt == "mp4":
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-maxrate", settings["videoBitrate"],
            "-bufsize", str(int(settings["videoBitrate"].replace("k", "")) * 2) + "k"
                if "k" in settings["videoBitrate"] else settings["videoBitrate"],
            "-pix_fmt", "yuv420p",
        ])
    elif fmt == "webm":
        cmd.extend([
            "-c:v", "libvpx-vp9",
            "-crf", "31",
            "-b:v", "0",
            "-deadline", "good",
            "-cpu-used", "2",
            "-row-mt", "1",
        ])

    if session["audio_path"] and os.path.exists(session["audio_path"]):
        cmd.extend([
            "-c:a", "aac" if fmt == "mp4" else "libopus",
            "-b:a", settings["audioBitrate"],
        ])

    cmd.extend(["-movflags", "+faststart", output_path])

    logger.info(f"[OpenReel Render] Session {render_id}: starting FFmpeg encode ({session['frame_count']} frames)")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[-500:]
            logger.error(f"[OpenReel Render] FFmpeg error: {err_msg}")
            session["status"] = "error"
            session["error"] = err_msg
            return web.json_response({"error": "FFmpeg encoding failed", "details": err_msg}, status=500)

        session["status"] = "uploading"
        session["progress"] = 0.95

        # Copy to ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        dest_path = os.path.join(input_dir, output_filename)
        shutil.copy2(output_path, dest_path)

        session["status"] = "complete"
        session["progress"] = 1.0
        session["output_path"] = dest_path

        # Write marker file so Unpack node can find the last backend export
        _write_export_marker(output_filename)

        file_size = os.path.getsize(dest_path)
        logger.info(f"[OpenReel Render] Session {render_id}: complete -> {output_filename} ({file_size} bytes)")

        return web.json_response({
            "status": "complete",
            "filename": output_filename,
            "size": file_size,
        })
    except Exception as e:
        session["status"] = "error"
        session["error"] = str(e)
        logger.error(f"[OpenReel Render] Session {render_id}: exception: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/was/openreel_video/render_direct")
async def render_direct(request):
    """
    Direct render for simple projects: receives a raw media file + settings
    and uses native FFmpeg to re-encode/trim. No frame-by-frame needed.

    Form fields:
      - media: file (the source video)
      - settings: JSON string with { width, height, frameRate, format,
        videoBitrate, audioBitrate, startTime?, endTime?, speed? }
    """
    try:
        reader = await request.multipart()
    except Exception:
        return web.json_response({"error": "Expected multipart data"}, status=400)

    media_data = None
    settings = None

    async for part in reader:
        if part.name == "media":
            media_data = await part.read(decode=False)
        elif part.name == "settings":
            settings = json.loads(await part.text())

    if not media_data or not settings:
        return web.json_response({"error": "Missing media or settings"}, status=400)

    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        return web.json_response({"error": "FFmpeg not found"}, status=500)

    render_dir = tempfile.mkdtemp(prefix="was_openreel_direct_")
    input_path = os.path.join(render_dir, "input")
    fmt = settings.get("format", "mp4")
    render_id = uuid.uuid4().hex[:16]
    output_filename = f"openreel_render_{render_id}.{fmt}"
    output_path = os.path.join(render_dir, output_filename)

    with open(input_path, "wb") as f:
        f.write(media_data)

    cmd = [ffmpeg_path, "-y"]

    start_time = settings.get("startTime", 0)
    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", input_path])

    end_time = settings.get("endTime")
    if end_time is not None and end_time > start_time:
        cmd.extend(["-t", str(end_time - start_time)])

    cmd.extend(["-threads", "0"])

    speed = settings.get("speed", 1)
    width = settings.get("width", 1920)
    height = settings.get("height", 1080)
    frame_rate = settings.get("frameRate", 30)
    video_bitrate = settings.get("videoBitrate", "12000k")
    audio_bitrate = settings.get("audioBitrate", "192k")

    needs_reencode = speed != 1

    if needs_reencode:
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={frame_rate}"
        if speed != 1 and speed > 0:
            video_speed = 1 / speed
            audio_speed = max(0.5, min(2.0, speed))
            cmd.extend([
                "-filter_complex",
                f"[0:v]{scale_filter},setpts={video_speed}*PTS[v];[0:a]atempo={audio_speed}[a]",
                "-map", "[v]", "-map", "[a]",
            ])
        else:
            cmd.extend(["-vf", scale_filter])
    else:
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={frame_rate}"
        cmd.extend(["-vf", scale_filter])

    if fmt == "mp4":
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-maxrate", video_bitrate,
            "-bufsize", str(int(video_bitrate.replace("k", "")) * 2) + "k"
                if "k" in video_bitrate else video_bitrate,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ])
    elif fmt == "webm":
        cmd.extend([
            "-c:v", "libvpx-vp9",
            "-crf", "31", "-b:v", "0",
            "-deadline", "good", "-cpu-used", "2", "-row-mt", "1",
            "-c:a", "libopus",
            "-b:a", audio_bitrate,
        ])

    cmd.extend(["-movflags", "+faststart", output_path])

    logger.info(f"[OpenReel Direct Render] {render_id}: {width}x{height} @ {frame_rate}fps, speed={speed}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[-500:]
            logger.error(f"[OpenReel Direct Render] FFmpeg error: {err_msg}")
            return web.json_response({"error": "FFmpeg encoding failed", "details": err_msg}, status=500)

        # Copy to ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        dest_path = os.path.join(input_dir, output_filename)
        shutil.copy2(output_path, dest_path)

        # Write marker file so Unpack node can find the last backend export
        _write_export_marker(output_filename)

        file_size = os.path.getsize(dest_path)
        logger.info(f"[OpenReel Direct Render] {render_id}: complete -> {output_filename} ({file_size} bytes)")

        # Cleanup render dir
        shutil.rmtree(render_dir, ignore_errors=True)

        return web.json_response({
            "status": "complete",
            "filename": output_filename,
            "size": file_size,
        })
    except Exception as e:
        shutil.rmtree(render_dir, ignore_errors=True)
        logger.error(f"[OpenReel Direct Render] {render_id}: exception: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/was/openreel_video/render_status")
async def render_status(request):
    """Poll the status/progress of a render session."""
    render_id = request.query.get("render_id", "")
    session = _render_sessions.get(render_id)
    if not session:
        return web.json_response({"error": "Unknown render session"}, status=404)

    return web.json_response({
        "status": session["status"],
        "progress": session["progress"],
        "frames_received": session["frame_count"],
        "total_frames": session["total_frames"],
        "error": session.get("error"),
        "filename": os.path.basename(session["output_path"]) if session.get("output_path") else None,
    })


@PromptServer.instance.routes.post("/was/openreel_video/set_export_marker")
async def set_export_marker(request):
    """Allow the frontend to set the export marker after browser-side exports."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    filename = data.get("filename", "")
    if not filename:
        return web.json_response({"error": "Missing filename"}, status=400)

    _write_export_marker(filename)
    return web.json_response({"status": "ok"})


def _write_export_marker(filename: str):
    """Write the last backend export filename to a marker file in temp dir.

    The CV_OpenReelUnpack node checks this marker as a fallback when the
    Content Viewer's view_state cache is invalidated (e.g. Bundle Video
    re-executes with a new session_id, changing the input hash).
    """
    try:
        marker_path = os.path.join(
            folder_paths.get_temp_directory(), "was_openreel_last_export.txt"
        )
        with open(marker_path, "w") as f:
            f.write(filename)
        logger.info(f"[OpenReel Render] Export marker written: {filename}")
    except Exception as e:
        logger.warning(f"[OpenReel Render] Failed to write export marker: {e}")


# Export sessions dict so nodes can access it
def get_sessions():
    """Get the sessions dictionary for use by nodes."""
    return _openreel_sessions


logger.info("[OpenReel Video] API routes registered")
