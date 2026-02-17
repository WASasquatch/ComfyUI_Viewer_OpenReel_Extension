# Installation Guide

## Quick Start

### Prerequisites

1. **ComfyUI** installed and working
2. **ComfyUI_Viewer** extension installed

### Installation Steps

#### 1. Install ComfyUI_Viewer (if not already installed)

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/WASasquatch/ComfyUI_Viewer.git
```

#### 2. Clone this Extension

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/WASasquatch/ComfyUI_Viewer_OpenReel_Extension.git
```

#### 3. Download the Built OpenReel App

The OpenReel app is **not included** in this repository to keep it lightweight. Download it from releases:

1. Go to: https://github.com/WASasquatch/openreel-video-comfyui/releases
2. Download the latest `openreel-app-vX.X.X.zip` (approximately 3.7 MB)
3. Extract the ZIP file
4. Copy all contents into `ComfyUI_Viewer_OpenReel_Extension/apps/openreel_app/`

**Expected directory structure after extraction:**

```
ComfyUI/custom_nodes/ComfyUI_Viewer_OpenReel_Extension/
├── apps/
│   └── openreel_app/
│       ├── assets/              # JavaScript chunks, CSS, WASM files
│       ├── workers/             # Web workers
│       ├── index.html           # Main HTML file
│       ├── manifest.json        # PWA manifest
│       ├── sw.js                # Service worker
│       ├── favicon.svg          # Icon
│       ├── _headers             # Netlify headers (optional)
│       └── _redirects           # Netlify redirects (optional)
├── modules/
├── nodes/
├── web/
└── README.md
```

#### 4. Restart ComfyUI

Restart ComfyUI completely (not just refresh the browser).

#### 5. Verify Installation

After restart, check for these nodes in the node menu under `WAS/View`:
- `CV OpenReel Bundle Video`
- `CV OpenReel Import Video`
- `CV OpenReel Unpack`

## Troubleshooting

### Nodes Don't Appear

**Check the console output:**
```
# Look for errors like:
# - "OpenReel app not found"
# - "Failed to load openreel_video_parser"
```

**Common issues:**
1. **Missing built app** - Make sure you downloaded and extracted the app to `apps/openreel_app/`
2. **Wrong directory** - Verify the files are in `openreel_app/`, not `openreel_app/openreel_app/`
3. **ComfyUI_Viewer not installed** - This extension requires ComfyUI_Viewer to be installed first

### OpenReel Doesn't Load in Viewer

**Check browser console (F12):**
- Look for 404 errors on `/was/openreel_video/app/` routes
- Check for CORS or module loading errors

**Solutions:**
1. Hard refresh the browser (Ctrl+Shift+R or Cmd+Shift+R)
2. Clear browser cache
3. Verify all files were extracted correctly
4. Check file permissions (should be readable)

### Video Import Fails

**Check:**
1. Video file is in ComfyUI's `input/` directory
2. Video format is supported (MP4, WebM, MOV, AVI)
3. PyAV is installed (`pip install av>=14.2.0`)

### Export Doesn't Work

**Check:**
1. You clicked "SEND TO OUTPUT" in OpenReel
2. You're using a Pause node in your workflow
3. You canceled and re-ran the workflow (not clicked "Continue")

## Manual Build (Advanced)

If you want to build the OpenReel app yourself:

```bash
# Clone the source repository
git clone https://github.com/WASasquatch/openreel-video-comfyui.git
cd openreel-video-comfyui

# Install dependencies
pnpm install

# Build
pnpm build

# Copy to extension
rm -rf ../ComfyUI_Viewer_OpenReel_Extension/apps/openreel_app/assets
cp -r apps/web/dist/* ../ComfyUI_Viewer_OpenReel_Extension/apps/openreel_app/
```

## Updating

### Update Extension Code

```bash
cd ComfyUI/custom_nodes/ComfyUI_Viewer_OpenReel_Extension
git pull
```

### Update OpenReel App

1. Check for new releases: https://github.com/WASasquatch/openreel-video-comfyui/releases
2. Download the latest `openreel-app-vX.X.X.zip`
3. **Delete old app files first:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI_Viewer_OpenReel_Extension/apps/openreel_app
   rm -rf assets workers *.html *.js *.json *.svg _headers _redirects
   ```
4. Extract new files to `apps/openreel_app/`
5. Restart ComfyUI

## Uninstallation

```bash
cd ComfyUI/custom_nodes/
rm -rf ComfyUI_Viewer_OpenReel_Extension
```

Then restart ComfyUI.

## Support

- **Extension issues**: https://github.com/WASasquatch/ComfyUI_Viewer_OpenReel_Extension/issues
- **OpenReel app issues**: https://github.com/WASasquatch/openreel-video-comfyui/issues
- **ComfyUI_Viewer issues**: https://github.com/WASasquatch/ComfyUI_Viewer/issues
