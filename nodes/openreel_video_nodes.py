import os
import json
import uuid
import hashlib
import shutil
import logging

import av
import torch
import numpy as np
import folder_paths

logger = logging.getLogger("WAS.OpenReel")

# Import sessions from routes module
# The routes module manages the session dictionary and API endpoints
try:
    import sys
    routes_path = os.path.join(os.path.dirname(__file__), "..", "routes")
    if routes_path not in sys.path:
        sys.path.insert(0, routes_path)
    from openreel_video_routes import get_sessions
    _openreel_sessions = get_sessions()
except ImportError:
    # Fallback if routes module not loaded yet
    _openreel_sessions = {}


# ============================================================================
# Node: CV OpenReel Bundle Video
# ============================================================================


class CV_OpenReelBundleVideo:
    """
    Bundles IMAGE frames and optional AUDIO from ComfyUI workflow into a video
    for editing in OpenReel.

    This allows you to edit videos generated within ComfyUI (e.g., from AnimateDiff,
    SVD, or any frame generation workflow) rather than loading external video files.

    Flow: [Generate Frames] -> [CV OpenReel Bundle Video] -> [Content Viewer] -> [CV OpenReel Unpack]
    """

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1},
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openreel_options",)
    FUNCTION = "bundle_video"
    CATEGORY = "WAS/View"

    def bundle_video(self, images, fps: float, audio=None) -> tuple:
        """
        Encode IMAGE frames and optional AUDIO to a video file, store in temp,
        and return tagged JSON for the Content Viewer parser.
        """
        session_id = str(uuid.uuid4())[:8]
        temp_dir = folder_paths.get_temp_directory()
        session_dir = os.path.join(temp_dir, f"was_openreel_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Clean up old sessions
        self._cleanup_old_sessions(temp_dir)

        # Get video dimensions from first frame
        frame_count = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        # Encode frames to video
        temp_video_path = os.path.join(session_dir, "video.mp4")

        # Prepare audio data before opening the container so all streams
        # can be created upfront (PyAV requires streams before any muxing)
        audio_waveform = None
        sample_rate = 44100
        audio_np = None
        audio_channels = 1

        if audio is not None:
            try:
                audio_waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate", 44100)
                if audio_waveform is not None:
                    audio_np = audio_waveform.cpu().numpy()
                    if len(audio_np.shape) == 3:
                        audio_np = audio_np[0]  # Remove batch dimension
                    audio_channels = (
                        audio_np.shape[0] if len(audio_np.shape) > 1 else 1
                    )
            except Exception as e:
                logger.warning(f"[OpenReel Bundle] Failed to prepare audio: {e}")
                audio_np = None

        with av.open(temp_video_path, mode="w") as container:
            # Create video stream
            stream = container.add_stream("h264", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}

            # Create audio stream upfront if audio is available
            audio_stream = None
            if audio_np is not None:
                try:
                    audio_layout = "stereo" if audio_channels == 2 else "mono"
                    audio_stream = container.add_stream(
                        "aac", rate=sample_rate, layout=audio_layout
                    )
                except Exception as e:
                    logger.warning(f"[OpenReel Bundle] Failed to create audio stream: {e}")
                    audio_stream = None

            # Encode video frames
            for i in range(frame_count):
                # Convert tensor to numpy array (0-255 uint8)
                frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)

                # Create VideoFrame
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")

                # Encode and write
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush remaining video packets
            for packet in stream.encode():
                container.mux(packet)

            # Encode audio if stream was created
            if audio_stream is not None and audio_np is not None:
                try:
                    audio_frame = av.AudioFrame.from_ndarray(
                        audio_np,
                        format="fltp",
                        layout="stereo" if audio_channels == 2 else "mono",
                    )
                    audio_frame.sample_rate = sample_rate

                    for packet in audio_stream.encode(audio_frame):
                        container.mux(packet)

                    # Flush audio
                    for packet in audio_stream.encode():
                        container.mux(packet)
                except Exception as e:
                    logger.warning(f"[OpenReel Bundle] Failed to encode audio: {e}")

        # Register session
        _openreel_sessions[session_id] = {
            "video_path": temp_video_path,
            "audio_path": None,
        }

        # Build the options JSON
        duration = frame_count / fps
        options = {
            "type": "openreel_video",
            "session_id": session_id,
            "has_video": True,
            "has_audio": audio is not None,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
        }

        result = self.OPENREEL_MARKER + json.dumps(options)
        return (result,)

    def _cleanup_old_sessions(self, temp_dir: str):
        """Remove old OpenReel temp directories, keeping only the most recent."""
        try:
            openreel_dirs = []
            for name in os.listdir(temp_dir):
                if name.startswith("was_openreel_"):
                    full_path = os.path.join(temp_dir, name)
                    if os.path.isdir(full_path):
                        openreel_dirs.append((full_path, os.path.getmtime(full_path)))

            # Sort by modification time, keep only the 10 most recent
            openreel_dirs.sort(key=lambda x: x[1], reverse=True)
            for dir_path, _ in openreel_dirs[10:]:
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, images, fps, audio=None):
        h = hashlib.sha256()
        h.update(f"{images.shape}_{images.dtype}_{fps}".encode())
        # Sample a few pixels for a fast content fingerprint
        flat = images.reshape(-1)
        stride = max(1, len(flat) // 64)
        for i in range(0, len(flat), stride):
            h.update(str(flat[i].item()).encode())
        if audio is not None:
            wf = audio.get("waveform")
            if wf is not None:
                h.update(f"{wf.shape}".encode())
                af = wf.reshape(-1)
                astride = max(1, len(af) // 32)
                for i in range(0, len(af), astride):
                    h.update(str(af[i].item()).encode())
        return h.hexdigest()


# ============================================================================
# Node: CV OpenReel Bundle Images
# ============================================================================


class CV_OpenReelBundleImages:
    """
    Bundles IMAGE(s) from ComfyUI workflow as individual image files
    for editing in OpenReel.

    Unlike Bundle Video (which encodes frames into a single video),
    this node saves each image as a separate PNG so OpenReel can import
    them as individual image clips on the timeline.

    Flow: [Generate Images] -> [CV OpenReel Bundle Images] -> [Content Viewer] -> [CV OpenReel Unpack]
    """

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openreel_options",)
    FUNCTION = "bundle_images"
    CATEGORY = "WAS/View"

    def bundle_images(self, images) -> tuple:
        """
        Save IMAGE tensors as individual PNG files, store in temp,
        and return tagged JSON for the Content Viewer parser.
        """
        session_id = str(uuid.uuid4())[:8]
        temp_dir = folder_paths.get_temp_directory()
        session_dir = os.path.join(temp_dir, f"was_openreel_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Clean up old sessions
        self._cleanup_old_sessions(temp_dir)

        image_count = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        # Save each image as PNG
        from PIL import Image

        image_filenames = []
        for i in range(image_count):
            frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            filename = f"image_{i:04d}.png"
            filepath = os.path.join(session_dir, filename)
            Image.fromarray(frame_np).save(filepath, format="PNG")
            image_filenames.append(filename)

        # Register session
        _openreel_sessions[session_id] = {
            "video_path": None,
            "audio_path": None,
            "image_dir": session_dir,
            "image_filenames": image_filenames,
        }

        # Build the options JSON
        options = {
            "type": "openreel_images",
            "session_id": session_id,
            "has_video": False,
            "has_audio": False,
            "width": width,
            "height": height,
            "image_count": image_count,
            "image_filenames": image_filenames,
        }

        result = self.OPENREEL_MARKER + json.dumps(options)
        return (result,)

    def _cleanup_old_sessions(self, temp_dir: str):
        """Remove old OpenReel temp directories, keeping only the most recent."""
        try:
            openreel_dirs = []
            for name in os.listdir(temp_dir):
                if name.startswith("was_openreel_"):
                    full_path = os.path.join(temp_dir, name)
                    if os.path.isdir(full_path):
                        openreel_dirs.append((full_path, os.path.getmtime(full_path)))

            # Sort by modification time, keep only the 10 most recent
            openreel_dirs.sort(key=lambda x: x[1], reverse=True)
            for dir_path, _ in openreel_dirs[10:]:
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, images):
        h = hashlib.sha256()
        h.update(f"{images.shape}_{images.dtype}".encode())
        flat = images.reshape(-1)
        stride = max(1, len(flat) // 64)
        for i in range(0, len(flat), stride):
            h.update(str(flat[i].item()).encode())
        return h.hexdigest()


# ============================================================================
# Node: CV OpenReel Bundle Audio
# ============================================================================


class CV_OpenReelBundleAudio:
    """
    Bundles AUDIO from ComfyUI workflow for editing in OpenReel.

    Saves the audio as a WAV file so OpenReel can import it as an
    audio clip on the timeline.

    Flow: [Audio Source] -> [CV OpenReel Bundle Audio] -> [Content Viewer]
    """

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openreel_options",)
    FUNCTION = "bundle_audio"
    CATEGORY = "WAS/View"

    def bundle_audio(self, audio) -> tuple:
        """
        Save AUDIO tensor as a WAV file, store in temp,
        and return tagged JSON for the Content Viewer parser.
        """
        session_id = str(uuid.uuid4())[:8]
        temp_dir = folder_paths.get_temp_directory()
        session_dir = os.path.join(temp_dir, f"was_openreel_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        self._cleanup_old_sessions(temp_dir)

        audio_waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if audio_waveform is None:
            raise ValueError("Audio input has no waveform data")

        import wave

        audio_np = audio_waveform.cpu().numpy()
        if len(audio_np.shape) == 3:
            audio_np = audio_np[0]  # Remove batch dimension

        channels = audio_np.shape[0] if len(audio_np.shape) > 1 else 1

        # Interleave channels: (channels, samples) -> (samples, channels)
        if len(audio_np.shape) > 1 and channels > 1:
            audio_interleaved = audio_np.T
        else:
            audio_interleaved = audio_np.flatten()

        # Convert float [-1, 1] to int16
        audio_int16 = (np.clip(audio_interleaved, -1.0, 1.0) * 32767).astype(np.int16)

        # Save as WAV using Python's wave module
        audio_path = os.path.join(session_dir, "audio.wav")
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Calculate duration from waveform length
        num_samples = audio_np.shape[-1] if len(audio_np.shape) > 1 else len(audio_np)
        duration = num_samples / sample_rate

        # Register session
        _openreel_sessions[session_id] = {
            "video_path": None,
            "audio_path": audio_path,
        }

        options = {
            "type": "openreel_audio",
            "session_id": session_id,
            "has_video": False,
            "has_audio": True,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
        }

        result = self.OPENREEL_MARKER + json.dumps(options)
        return (result,)

    def _cleanup_old_sessions(self, temp_dir: str):
        """Remove old OpenReel temp directories, keeping only the most recent."""
        try:
            openreel_dirs = []
            for name in os.listdir(temp_dir):
                if name.startswith("was_openreel_"):
                    full_path = os.path.join(temp_dir, name)
                    if os.path.isdir(full_path):
                        openreel_dirs.append((full_path, os.path.getmtime(full_path)))

            # Sort by modification time, keep only the 10 most recent
            openreel_dirs.sort(key=lambda x: x[1], reverse=True)
            for dir_path, _ in openreel_dirs[10:]:
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, audio):
        h = hashlib.sha256()
        wf = audio.get("waveform") if audio else None
        sr = audio.get("sample_rate", 0) if audio else 0
        if wf is not None:
            h.update(f"{wf.shape}_{wf.dtype}_{sr}".encode())
            flat = wf.reshape(-1)
            stride = max(1, len(flat) // 64)
            for i in range(0, len(flat), stride):
                h.update(str(flat[i].item()).encode())
        else:
            h.update(b"no_audio")
        return h.hexdigest()


# ============================================================================
# Node: CV OpenReel Bundle Combine
# ============================================================================


class CV_OpenReelBundleCombine:
    """
    Combines outputs from multiple Bundle nodes (Video, Images, Audio)
    into a single tagged JSON so OpenReel imports all assets at once.

    Connect the openreel_options outputs from any Bundle nodes to the
    inputs of this node. All media will be imported together.

    Flow: [Bundle Video] ─┐
          [Bundle Images] ─┤─> [CV OpenReel Bundle Combine] -> [Content Viewer]
          [Bundle Audio]  ─┘
    """

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_a": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "First bundle output (from any Bundle node)",
                    },
                ),
            },
            "optional": {
                "bundle_b": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Second bundle output (optional)",
                    },
                ),
                "bundle_c": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Third bundle output (optional)",
                    },
                ),
                "bundle_d": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Fourth bundle output (optional)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openreel_options",)
    FUNCTION = "combine_bundles"
    CATEGORY = "WAS/View"

    def combine_bundles(
        self, bundle_a, bundle_b=None, bundle_c=None, bundle_d=None
    ) -> tuple:
        """
        Parse each bundle input, collect all media entries,
        and produce a single combined tagged JSON.
        """
        bundles = [bundle_a]
        if bundle_b is not None:
            bundles.append(bundle_b)
        if bundle_c is not None:
            bundles.append(bundle_c)
        if bundle_d is not None:
            bundles.append(bundle_d)

        entries = []
        for raw in bundles:
            if not isinstance(raw, str) or not raw.startswith(self.OPENREEL_MARKER):
                continue
            try:
                data = json.loads(raw[len(self.OPENREEL_MARKER):])
                entries.append(data)
            except (json.JSONDecodeError, AttributeError):
                continue

        if not entries:
            raise ValueError("No valid bundle inputs provided")

        # Calculate total duration as the max across all entries
        total_duration = max(e.get("duration", 0) for e in entries)

        options = {
            "type": "openreel_combined",
            "entries": entries,
            "entry_count": len(entries),
            "duration": total_duration,
        }

        result = self.OPENREEL_MARKER + json.dumps(options)
        return (result,)

    @classmethod
    def IS_CHANGED(cls, bundle_a, bundle_b=None, bundle_c=None, bundle_d=None):
        h = hashlib.sha256()
        for val in (bundle_a, bundle_b, bundle_c, bundle_d):
            h.update((val or "").encode())
        return h.hexdigest()


# ============================================================================
# Node: CV OpenReel Import Video
# ============================================================================


class CV_OpenReelImportVideo:
    """
    Loads a video file, copies it to a temp session directory, and outputs
    a tagged JSON string (openreel_options) for the Content Viewer parser.

    The Content Viewer will display the OpenReel editor with this video loaded.

    Flow: [CV OpenReel Import Video] -> [Content Viewer] -> [CV OpenReel Unpack]
    """

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openreel_options",)
    FUNCTION = "import_video"
    CATEGORY = "WAS/View"

    def import_video(self, video: str) -> tuple:
        """
        Load a video file, probe its metadata, copy to temp, and return
        tagged JSON for the Content Viewer parser.
        """
        video_path = folder_paths.get_annotated_filepath(video)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        session_id = str(uuid.uuid4())[:8]
        temp_dir = folder_paths.get_temp_directory()
        session_dir = os.path.join(temp_dir, f"was_openreel_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Clean up old sessions
        self._cleanup_old_sessions(temp_dir)

        # Probe the source video for metadata
        has_video = False
        has_audio = False
        width = 0
        height = 0
        frame_count = 0
        fps = 24.0
        duration = 0.0

        with av.open(video_path, mode="r") as container:
            for stream in container.streams:
                if stream.type == "video" and not has_video:
                    has_video = True
                    width = stream.codec_context.width
                    height = stream.codec_context.height
                    fps = float(stream.average_rate) if stream.average_rate else 24.0
                    frame_count = stream.frames or 0
                    if stream.duration and stream.time_base:
                        duration = float(stream.duration * stream.time_base)
                    elif frame_count and fps:
                        duration = frame_count / fps
                elif stream.type == "audio" and not has_audio:
                    has_audio = True

            # If frame_count wasn't available from metadata, count manually
            if has_video and frame_count == 0:
                container.seek(0)
                frame_count = sum(1 for _ in container.decode(video=0))
                if frame_count and fps:
                    duration = frame_count / fps

        # Copy the source video to the session temp directory
        temp_video_path = os.path.join(session_dir, "video.mp4")
        shutil.copy2(video_path, temp_video_path)

        # Register session
        _openreel_sessions[session_id] = {
            "video_path": temp_video_path,
            "audio_path": None,
        }

        # Build the options JSON
        options = {
            "type": "openreel_video",
            "session_id": session_id,
            "has_video": has_video,
            "has_audio": has_audio,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
        }

        result = self.OPENREEL_MARKER + json.dumps(options)
        return (result,)

    def _cleanup_old_sessions(self, temp_dir: str):
        """Remove old OpenReel temp directories, keeping only the most recent."""
        try:
            openreel_dirs = []
            for name in os.listdir(temp_dir):
                if name.startswith("was_openreel_"):
                    full_path = os.path.join(temp_dir, name)
                    if os.path.isdir(full_path):
                        openreel_dirs.append((full_path, os.path.getmtime(full_path)))

            # Sort by modification time, keep only the 10 most recent
            openreel_dirs.sort(key=lambda x: x[1], reverse=True)
            for dir_path, _ in openreel_dirs[10:]:
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, video):
        video_path = folder_paths.get_annotated_filepath(video)
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


# ============================================================================
# Node: CV OpenReel Unpack
# ============================================================================


class CV_OpenReelUnpack:
    """
    Unpacks an edited video exported from OpenReel via the Content Viewer.

    The Content Viewer outputs the filename of the edited video that was
    uploaded to ComfyUI's input directory. This node loads that video file
    and decomposes it into IMAGE frames, AUDIO, and fps for downstream use.

    Flow: [Content Viewer] -> [CV OpenReel Unpack] -> IMAGE / AUDIO / fps
    """

    OUTPUT_MARKER = "$WAS_OPENREEL_VIDEO_OUTPUT$"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_filename": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Video filename from Content Viewer output (edited OpenReel export)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "unpack_video"
    CATEGORY = "WAS/View"

    OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$"

    def unpack_video(self, video_filename: str) -> tuple:
        """Load the edited video file and decompose into frames + audio."""
        # Strip output marker if present (in case raw parser output is passed)
        filename = video_filename
        if filename.startswith(self.OUTPUT_MARKER):
            try:
                data = json.loads(filename[len(self.OUTPUT_MARKER) :])
                filename = data.get("filename", filename)
            except (json.JSONDecodeError, AttributeError):
                pass

        # If we received the INPUT marker (no export yet), fall back to the
        # original source video from the session's temp directory.
        if filename.startswith(self.OPENREEL_MARKER):
            try:
                data = json.loads(filename[len(self.OPENREEL_MARKER) :])
                fps_fallback = data.get("fps", 24.0)

                # Check for a backend render export marker first — this is
                # written by render_finalize / render_direct and persists
                # across workflow re-runs even when the input hash changes.
                export_result = self._check_export_marker()
                if export_result is not None:
                    return export_result

                session_id = data.get("session_id", "")
                if session_id and session_id in _openreel_sessions:
                    session = _openreel_sessions[session_id]
                    video_path = session.get("video_path", "")
                    if video_path and os.path.exists(video_path):
                        return self._decode_video(video_path)
                # Session not found or expired — return placeholder
                logger.warning(
                    f"[OpenReel Unpack] No edited video yet (session: {data.get('session_id', '?')}). "
                    "Export a video from OpenReel first, or re-run the workflow."
                )
                return (torch.zeros(1, 64, 64, 3), None, fps_fallback)
            except (json.JSONDecodeError, AttributeError):
                return (torch.zeros(1, 64, 64, 3), None, 24.0)

        # Resolve the file path in ComfyUI's input directory
        video_path = folder_paths.get_annotated_filepath(filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Edited video not found: {filename} (resolved to {video_path})"
            )

        return self._decode_video(video_path)

    def _check_export_marker(self):
        """Check for a backend render export marker file.

        Returns the decoded video tuple if a valid export exists, else None.
        """
        try:
            marker_path = os.path.join(
                folder_paths.get_temp_directory(), "was_openreel_last_export.txt"
            )
            if not os.path.exists(marker_path):
                return None
            with open(marker_path, "r") as f:
                export_filename = f.read().strip()
            if not export_filename:
                return None
            video_path = folder_paths.get_annotated_filepath(export_filename)
            if os.path.exists(video_path):
                logger.info(
                    f"[OpenReel Unpack] Using backend export: {export_filename}"
                )
                return self._decode_video(video_path)
        except Exception as e:
            logger.warning(f"[OpenReel Unpack] Export marker check failed: {e}")
        return None

    def _decode_video(self, video_path: str) -> tuple:
        """Decode a video file into IMAGE frames, AUDIO, and fps."""
        images = None
        audio_output = None
        fps_value = 24.0

        with av.open(video_path, mode="r") as container:
            # Extract video frames
            frames = []
            video_stream = None
            for stream in container.streams:
                if stream.type == "video":
                    video_stream = stream
                    break

            if video_stream:
                fps_value = (
                    float(video_stream.average_rate)
                    if video_stream.average_rate
                    else 24.0
                )
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="rgb24")
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    frames.append(img_tensor)

            if frames:
                images = torch.stack(frames)
            else:
                images = torch.zeros(1, 64, 64, 3)

            # Extract audio
            try:
                container.seek(0)
                for stream in container.streams:
                    if stream.type != "audio":
                        continue

                    audio_frames = []
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            audio_frames.append(frame.to_ndarray())

                    if audio_frames:
                        audio_data = np.concatenate(audio_frames, axis=1)
                        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                        audio_output = {
                            "waveform": audio_tensor,
                            "sample_rate": (
                                int(stream.sample_rate) if stream.sample_rate else 44100
                            ),
                        }
                    break
            except Exception:
                pass

        return (images, audio_output, fps_value)

    @classmethod
    def IS_CHANGED(cls, video_filename):
        filename = video_filename
        if filename.startswith(cls.OUTPUT_MARKER):
            try:
                data = json.loads(filename[len(cls.OUTPUT_MARKER) :])
                filename = data.get("filename", filename)
            except (json.JSONDecodeError, AttributeError):
                pass
        # Input marker means no export yet — always re-check
        if filename.startswith(cls.OPENREEL_MARKER):
            return float("nan")
        video_path = folder_paths.get_annotated_filepath(filename)
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "CV_OpenReelBundleVideo": CV_OpenReelBundleVideo,
    "CV_OpenReelBundleImages": CV_OpenReelBundleImages,
    "CV_OpenReelBundleAudio": CV_OpenReelBundleAudio,
    "CV_OpenReelBundleCombine": CV_OpenReelBundleCombine,
    "CV_OpenReelImportVideo": CV_OpenReelImportVideo,
    "CV_OpenReelUnpack": CV_OpenReelUnpack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_OpenReelBundleVideo": "CV OpenReel Bundle Video",
    "CV_OpenReelBundleImages": "CV OpenReel Bundle Images",
    "CV_OpenReelBundleAudio": "CV OpenReel Bundle Audio",
    "CV_OpenReelBundleCombine": "CV OpenReel Bundle Combine",
    "CV_OpenReelImportVideo": "CV OpenReel Import Video",
    "CV_OpenReelUnpack": "CV OpenReel Unpack",
}
