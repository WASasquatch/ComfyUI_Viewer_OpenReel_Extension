"""
OpenReel Video Parser for WAS Content Viewer.

Handles:
- INPUT: Tagged JSON from CV_OpenReelBundleVideo node
- Passes through video metadata and serve URLs for the frontend view
- OUTPUT: Video file path from OpenReel's "Send to Output" action
"""

import json
import hashlib

try:
    from .base_parser import BaseParser
except ImportError:
    import importlib.util
    import os

    _bp = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "ComfyUI_Viewer",
        "modules",
        "parsers",
        "base_parser.py",
    )
    if not os.path.exists(_bp):
        _bp = os.path.join(os.path.dirname(__file__), "base_parser.py")
    _spec = importlib.util.spec_from_file_location("base_parser", _bp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseParser = _mod.BaseParser


class OpenReelVideoParser(BaseParser):
    """OpenReel Video parser for video editing integration."""

    PARSER_NAME = "openreel_video"
    PARSER_PRIORITY = 108

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"
    OUTPUT_MARKER = "$WAS_OPENREEL_VIDEO_OUTPUT$"

    @classmethod
    def detect_input(cls, content) -> bool:
        """Check if content is OpenReel video options JSON."""
        if content is None:
            return False

        items = content if isinstance(content, (list, tuple)) else [content]

        for item in items:
            if isinstance(item, str) and item.startswith(cls.OPENREEL_MARKER):
                return True
        return False

    @classmethod
    def handle_input(cls, content, logger=None) -> dict:
        """
        Process OpenReel video options and pass through for the frontend view.

        The node has already encoded the video and stored it in temp.
        We just need to pass the metadata through to the frontend.
        """
        items = content if isinstance(content, (list, tuple)) else [content]

        openreel_content = None
        for item in items:
            if isinstance(item, str) and item.startswith(cls.OPENREEL_MARKER):
                openreel_content = item
                break

        if not openreel_content:
            return None

        try:
            options = json.loads(openreel_content[len(cls.OPENREEL_MARKER) :])
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"[OpenReel Video Parser] Invalid JSON: {e}")
            return None

        if options.get("type") != "openreel_video":
            return None

        session_id = options.get("session_id", "")

        # Build the display content with serve URLs for the frontend
        display_data = {
            "type": "openreel_video",
            "session_id": session_id,
            "has_video": options.get("has_video", False),
            "has_audio": options.get("has_audio", False),
            "width": options.get("width", 0),
            "height": options.get("height", 0),
            "frame_count": options.get("frame_count", 0),
            "fps": options.get("fps", 24.0),
            "duration": options.get("duration", 0.0),
            "video_url": (
                f"/was/openreel_video/serve?session_id={session_id}"
                if options.get("has_video")
                else None
            ),
            "audio_url": (
                f"/was/openreel_video/serve_audio?session_id={session_id}"
                if options.get("has_audio") and not options.get("has_video")
                else None
            ),
        }

        display_content = cls.OPENREEL_MARKER + json.dumps(display_data)

        # Content hash for caching
        content_hash = hashlib.md5(
            json.dumps(options, sort_keys=True).encode()
        ).hexdigest()

        return {
            "display_content": display_content,
            "output_values": [display_content],
            "content_hash": content_hash,
        }

    @classmethod
    def detect_output(cls, content: str) -> bool:
        """Check if content contains the OpenReel output marker."""
        if isinstance(content, str):
            return content.startswith(cls.OUTPUT_MARKER)
        return False

    @classmethod
    def parse_output(cls, content: str, logger=None) -> dict:
        """
        Parse output from the OpenReel frontend view.

        The frontend sends back the filename of the exported video
        that was uploaded to ComfyUI's input directory.
        """
        if not isinstance(content, str) or not content.startswith(cls.OUTPUT_MARKER):
            return None

        try:
            data = json.loads(content[len(cls.OUTPUT_MARKER) :])
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"[OpenReel Video Parser] Invalid output JSON: {e}")
            return None

        filename = data.get("filename", "")
        if not filename:
            if logger:
                logger.warning("[OpenReel Video Parser] No filename in output")
            return None

        content_hash = hashlib.md5(filename.encode()).hexdigest()

        return {
            "output_values": [filename],
            "display_text": f"OpenReel exported: {filename}",
            "content_hash": content_hash,
        }
