# ComfyUI Viewer - OpenReel Video Extension

Embeds a [modified fork of OpenReel Video](https://github.com/WASasquatch/openreel-video-comfyui) into ComfyUI_Viewer for interactive video editing within ComfyUI workflows.

## Features

- **Full-featured video editor** embedded in ComfyUI with timeline, effects, transitions, and text overlays
- **Seamless integration** with ComfyUI's theme system (dark/light mode sync)
- **Direct workflow integration** import videos, edit in OpenReel, export back to ComfyUI nodes
- **Standalone mode** launch OpenReel in a new tab for full-screen editing
- **No external dependencies** all processing happens locally in your browser

## Nodes

### CV OpenReel Bundle Video
Bundles IMAGE frames and optional AUDIO from ComfyUI workflows into a video for editing in OpenReel.

**Inputs:**
- `images` (IMAGE, required) — batch of frames from any ComfyUI video generation workflow
- `fps` (FLOAT, required) — frame rate (default 24.0, range 1.0-120.0)
- `audio` (AUDIO, optional) — audio track to include in the video

**Output:** `STRING` — tagged JSON for ComfyUI_Viewer's `content` input

**Use Case:** Edit videos generated within ComfyUI (e.g., from AnimateDiff, SVD, frame interpolation, etc.)

### CV OpenReel Import Video
Loads an external video file and outputs tagged JSON for the Content Viewer to display in OpenReel.

**Inputs:**
- `video` (COMBO) — video file from ComfyUI's `input/` directory (with upload button)

**Output:** `STRING` — tagged JSON for ComfyUI_Viewer's `content` input

**Use Case:** Edit existing video files from disk

### CV OpenReel Unpack
Unpacks an edited video exported from OpenReel into IMAGE frames, AUDIO, and fps.

**Inputs:**
- `video_filename` (STRING, force input) — filename from Content Viewer output

**Outputs:**
- `images` (IMAGE) — decoded frames as tensor batch
- `audio` (AUDIO) — extracted audio (or None if no audio track)
- `fps` (FLOAT) — frame rate

## Workflow Setup

**It's highly recommended to use a pause node** to prevent the workflow from continuing while you edit your video in OpenReel.

We recommend [ComfyUI-pause](https://github.com/wywywywy/ComfyUI-pause)

### Recommended Workflow Structure:

**For External Videos:**
```
[CV OpenReel Import Video] → [Content Viewer] → [Pause Node] → [CV OpenReel Unpack] → [Save Image/Video]
```

**For ComfyUI-Generated Videos:**
```
[Generate Frames] → [CV OpenReel Bundle Video] → [Content Viewer] → [Pause Node] → [CV OpenReel Unpack] → [Save Image/Video]
```

### How the Pause Workflow Works:

1. **First Run ("Edit Run")**: The workflow runs and pauses at the Pause node
   - Your video loads into OpenReel editor in the Content Viewer
   - The workflow is suspended, giving you time to edit
   - Edit your video (trim, effects, transitions, text, etc.)
   - Click **SEND TO OUTPUT** when finished editing

2. **Second Run ("Export Run")**: Cancel the workflow and run it again
   - Click **Cancel** button (do NOT click "Continue")
   - Run the workflow again
   - Your edited video will be unpacked and passed to downstream nodes
   - The output is ready for Save Image or further processing

**Important**: If you click "Continue" instead of "Cancel", the original input video will be passed through, not your edited version.

## Workflow Examples

### Workflow 1: Edit External Video Files
1. Add `CV OpenReel Import Video` node and select a video file
2. Connect its output to `WAS ComfyUI Viewer` → `content` input
3. Add a **Pause node** after Content Viewer
4. Add `CV OpenReel Unpack` node after the Pause node
5. Connect Content Viewer's output to CV OpenReel Unpack's `video_filename` input
6. **First run**: Workflow pauses, edit your video in OpenReel, click **SEND TO OUTPUT**
7. **Cancel workflow** and run again — edited video is unpacked as IMAGE/AUDIO/fps

### Workflow 2: Edit ComfyUI-Generated Videos
1. Generate frames using any ComfyUI workflow (AnimateDiff, SVD, frame interpolation, etc.)
2. Add `CV OpenReel Bundle Video` node
3. Connect IMAGE output → `images` input, set `fps`, optionally connect AUDIO
4. Connect Bundle node output to `WAS ComfyUI Viewer` → `content` input
5. Add a **Pause node** after Content Viewer
6. Add `CV OpenReel Unpack` node after the Pause node
7. Connect Content Viewer's output to CV OpenReel Unpack's `video_filename` input
8. **First run**: Workflow pauses, edit in OpenReel, click **SEND TO OUTPUT**
9. **Cancel workflow** and run again — edited frames/audio ready for further processing

### Advanced Usage
- **Launch in new tab**: Click the external link icon in embedded mode to open OpenReel in a standalone tab for full-screen editing
- **Theme sync**: OpenReel automatically matches ComfyUI's theme (dark/light mode)
- **Persistent exports**: Once exported, the video persists in the workflow until you export a new one or change the input video

## Known Issues

### Firefox Hard Refresh Bug
**Issue**: In Firefox, performing a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) may cause the OpenReel iframe to enter a reload loop.

**Workaround**: Use a **soft refresh** (Ctrl+R / Cmd+R or F5) instead. This issue does not occur in Chrome/Edge.

**Technical Details**: Firefox's aggressive cache invalidation on hard refresh causes module loading failures in the iframe context. This is a browser-specific behavior and does not affect normal workflow usage.

## Requirements

- ComfyUI with PyAV (`av>=14.2.0`, included by default in most ComfyUI distributions)
- [ComfyUI_Viewer](https://github.com/WASasquatch/ComfyUI_Viewer) extension (must be installed first)

## Installation

### Option 1: Install via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "OpenReel Video Extension"
3. Click Install
4. Restart ComfyUI

### Option 2: Manual Installation

1. **Install ComfyUI_Viewer** if not already installed:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/WASasquatch/ComfyUI_Viewer.git
   ```

2. **Clone this extension**:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/WASasquatch/ComfyUI_Viewer_OpenReel_Extension.git
   ```

3. **Download the built OpenReel app**:
   - Go to [Releases](https://github.com/WASasquatch/openreel-video-comfyui/releases)
   - Download the latest `openreel-app-vX.X.X.zip`
   - Extract the contents into `ComfyUI_Viewer_OpenReel_Extension/apps/openreel_app/`
   
   The directory structure should look like:
   ```
   ComfyUI_Viewer_OpenReel_Extension/
   └── apps/
       └── openreel_app/
           ├── assets/
           ├── workers/
           ├── index.html
           ├── manifest.json
           └── ...
   ```

4. **Restart ComfyUI**

5. The OpenReel nodes will appear in the node menu under `WAS/View`

### Verifying Installation

After restarting ComfyUI, you should see these nodes:
- `CV OpenReel Bundle Video`
- `CV OpenReel Import Video`
- `CV OpenReel Unpack`

If the nodes don't appear, check the ComfyUI console for errors.

## Architecture

This extension demonstrates ComfyUI_Viewer's `/app` functionality for embedding full web applications:

- **Frontend**: Modified OpenReel React app served from `/was/openreel_video/app/`
  - Source: [openreel-video-comfyui](https://github.com/WASasquatch/openreel-video-comfyui) (fork with ComfyUI integration)
  - Built app distributed via GitHub Releases (not included in repository)
- **Backend**: Python nodes handle video I/O and session management
- **Communication**: PostMessage API for iframe ↔ ComfyUI communication
- **Parser**: Custom parser (`openreel_video_parser.py`) handles input/output data flow

### Why Separate Repositories?

- **Extension repository** (this repo): Python nodes, parsers, and integration code (~200 KB)
- **App source repository** ([openreel-video-comfyui](https://github.com/WASasquatch/openreel-video-comfyui)): React app source code (~7 MB)
- **Built app**: Distributed as release artifacts (~3.7 MB compressed)

This separation keeps the extension repository lightweight while allowing contributors to build and modify the OpenReel app independently.

See ComfyUI_Viewer's README for details on creating your own viewer extensions with embedded apps.

## License

MIT — see [LICENSE](LICENSE)
