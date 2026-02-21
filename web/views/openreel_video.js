/**
 * OpenReel Video View - Embeds the OpenReel Video editor for video editing
 * 
 * Features:
 * - Loads OpenReel Video app in an iframe
 * - Auto-imports video from the Bundle node's temp video
 * - "Send to Output" uploads exported video to ComfyUI input directory
 * - Sends output filename back to view_state for the Import node
 */

import { BaseView } from "./base_view.js";

class OpenReelVideoView extends BaseView {
  static id = "openreel_video";
  static displayName = "OpenReel Video";
  static priority = 108;
  static isUI = true;

  static OPENREEL_MARKER = "$WAS_OPENREEL_VIDEO$";
  static OUTPUT_MARKER = "$WAS_OPENREEL_VIDEO_OUTPUT$";

  static usesBaseStyles() {
    return false;
  }

  /**
   * Request sandbox permissions for the outer Content Viewer iframe.
   * OpenReel needs allow-same-origin so its inner iframe can fetch
   * from the ComfyUI API endpoints (video serve, upload, etc).
   */
  static getSandboxAttributes() {
    return "allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-pointer-lock allow-downloads";
  }

  /**
   * Request blob URL loading for the outer iframe.
   * srcdoc iframes always have a null origin, so relative URLs like
   * /was/openreel_video/app/index.html won't resolve. Blob URLs
   * inherit the creator's origin when allow-same-origin is set.
   */
  static needsBlobUrl() {
    return true;
  }

  static getContentMarker() {
    return this.OPENREEL_MARKER;
  }

  /**
   * Get message types this view handles
   */
  static getMessageTypes() {
    return ["openreel-video-output"];
  }

  /**
   * Handle messages from the OpenReel iframe
   */
  static handleMessage(messageType, data, node, app, iframeSource) {
    if (messageType !== "openreel-video-output") return false;

    const filename = data?.filename;
    if (!filename) {
      console.warn("[OpenReel Video View] No filename in output message");
      return false;
    }

    const outputData = { filename: filename };
    const outputString = this.OUTPUT_MARKER + JSON.stringify(outputData);

    const viewStateWidget = node.widgets?.find(w => w.name === "view_state");
    if (viewStateWidget) {
      try {
        const viewState = JSON.parse(viewStateWidget.value || "{}");
        viewState.openreel_video_output = outputString;
        
        // Store the current input_hash so the backend can verify cache validity
        // The input_hash is set by the backend during execution (onExecuted)
        if (viewState._input_hash) {
          // Keep the existing input_hash - it's already set by onExecuted
        }
        
        viewStateWidget.value = JSON.stringify(viewState);
      } catch (e) {
        console.error("[OpenReel Video View] Failed to save view state:", e);
      }

      console.log("[OpenReel Video View] Output saved:", filename);
      node.setDirtyCanvas?.(true, true);
      return true;
    }

    console.warn("[OpenReel Video View] No view_state widget found");
    return false;
  }

  /**
   * Parse content string into OpenReel data object.
   */
  static _parseContent(content) {
    if (!content) return null;
    try {
      let jsonContent = content;
      if (content.startsWith(this.OPENREEL_MARKER)) {
        jsonContent = content.slice(this.OPENREEL_MARKER.length);
      }
      const parsed = JSON.parse(jsonContent);
      if (["openreel_video", "openreel_images", "openreel_audio", "openreel_combined"].includes(parsed.type)) return parsed;
    } catch {}
    return null;
  }

  /**
   * Build a stable OpenReel app URL with ONLY embedding + theme params.
   * Video/session data is sent separately via postMessage so the iframe
   * never needs to be reloaded when content changes.
   */
  static _buildAppUrl(theme) {
    const origin = typeof window !== 'undefined' ? window.location.origin : '';
    const appParams = new URLSearchParams({ embedded: 'true' });

    if (theme) {
      if (theme.bg) appParams.set('theme_bg', theme.bg);
      if (theme.fg) appParams.set('theme_fg', theme.fg);
      if (theme.border) appParams.set('theme_border', theme.border);
      if (theme.accent) appParams.set('theme_accent', theme.accent);

      if (typeof document !== 'undefined') {
        const rs = getComputedStyle(document.documentElement);
        const readVar = (name) => { const v = rs.getPropertyValue(name); return v ? v.trim() : ''; };
        const bgSecondary = readVar('--theme-bg-light') || readVar('--comfy-input-bg');
        const bgTertiary = readVar('--theme-bg-dark') || readVar('--comfy-menu-secondary-bg');
        const fgMuted = readVar('--theme-fg-muted') || readVar('--descrip-text');
        const inputBg = readVar('--theme-input-bg') || readVar('--comfy-input-bg');
        const scrollThumb = readVar('--theme-scrollbar-thumb') || readVar('--comfy-scrollbar-thumb');
        const scrollTrack = readVar('--theme-scrollbar-track') || readVar('--comfy-scrollbar-track');
        if (bgSecondary) appParams.set('theme_bg_secondary', bgSecondary);
        if (bgTertiary) appParams.set('theme_bg_tertiary', bgTertiary);
        if (fgMuted) appParams.set('theme_fg_muted', fgMuted);
        if (inputBg) appParams.set('theme_input_bg', inputBg);
        if (scrollThumb) appParams.set('theme_scrollbar_thumb', scrollThumb);
        if (scrollTrack) appParams.set('theme_scrollbar_track', scrollTrack);
      }
    }

    return `${origin}/was/openreel_video/app/index.html?${appParams.toString()}`;
  }

  /**
   * Provide a stable direct URL for the outer Content Viewer iframe.
   * The URL contains NO video/session params — those are sent via postMessage.
   * This ensures the iframe is loaded once and never reloaded.
   */
  static getDirectUrl(content, theme) {
    return this._buildAppUrl(theme);
  }

  /**
   * Build a postMessage payload to send video content to the already-loaded
   * OpenReel iframe.  Called by comfy_viewer.js when content changes.
   */
  static getContentMessage(content) {
    const data = this._parseContent(content);
    if (!data) return null;

    const origin = typeof window !== 'undefined' ? window.location.origin : '';

    if (data.type === 'openreel_images') {
      const msg = { type: 'comfyui-import-images' };
      if (data.image_urls && data.image_urls.length > 0) {
        msg.imageUrls = data.image_urls.map(url => origin + url);
      }
      if (data.session_id) msg.sessionId = data.session_id;
      if (data.duration_per_image) msg.durationPerImage = data.duration_per_image;
      return msg;
    }

    if (data.type === 'openreel_audio') {
      const msg = { type: 'comfyui-import-video' };
      if (data.audio_url) {
        msg.audioUrl = origin + data.audio_url;
      }
      if (data.session_id) msg.sessionId = data.session_id;
      return msg;
    }

    if (data.type === 'openreel_combined') {
      const entries = data.entries || [];
      const imports = [];
      for (const entry of entries) {
        if (entry.type === 'openreel_video') {
          const m = { type: 'comfyui-import-video' };
          if (entry.video_url) m.url = origin + entry.video_url;
          if (entry.audio_url) m.audioUrl = origin + entry.audio_url;
          if (entry.session_id) m.sessionId = entry.session_id;
          if (entry.fps) m.fps = entry.fps;
          imports.push(m);
        } else if (entry.type === 'openreel_images') {
          const m = { type: 'comfyui-import-images' };
          if (entry.image_urls) m.imageUrls = entry.image_urls.map(url => origin + url);
          if (entry.session_id) m.sessionId = entry.session_id;
          if (entry.duration_per_image) m.durationPerImage = entry.duration_per_image;
          imports.push(m);
        } else if (entry.type === 'openreel_audio') {
          const m = { type: 'comfyui-import-video' };
          if (entry.audio_url) m.audioUrl = origin + entry.audio_url;
          if (entry.session_id) m.sessionId = entry.session_id;
          imports.push(m);
        }
      }
      return { type: 'comfyui-import-combined', imports };
    }

    const msg = { type: 'comfyui-import-video' };

    if (data.has_video && data.video_url) {
      msg.url = origin + data.video_url;
    }
    if (data.has_audio && data.audio_url) {
      msg.audioUrl = origin + data.audio_url;
    }
    if (data.session_id) msg.sessionId = data.session_id;
    if (data.fps) msg.fps = data.fps;

    return msg;
  }

  /**
   * Detect if content is OpenReel video data
   */
  static detect(content) {
    try {
      let jsonContent = content;
      if (content.startsWith(this.OPENREEL_MARKER)) {
        jsonContent = content.slice(this.OPENREEL_MARKER.length);
      }
      const parsed = JSON.parse(jsonContent);
      if (["openreel_video", "openreel_images", "openreel_audio", "openreel_combined"].includes(parsed.type)) {
        return 208;
      }
    } catch {}
    return 0;
  }

  /**
   * Render the OpenReel Video editor
   */
  static render(content, theme) {
    let data;
    try {
      let jsonContent = content;
      if (content.startsWith(this.OPENREEL_MARKER)) {
        jsonContent = content.slice(this.OPENREEL_MARKER.length);
      }
      data = JSON.parse(jsonContent);
    } catch {
      return `<pre style="color: #ff6b6b; padding: 20px;">Invalid OpenReel video data</pre>`;
    }

    const sessionId = data.session_id || '';
    const hasVideo = data.has_video || false;
    const hasAudio = data.has_audio || false;
    const videoUrl = data.video_url || '';
    const audioUrl = data.audio_url || '';
    const width = data.width || 0;
    const height = data.height || 0;
    const fps = data.fps || 24;
    const duration = data.duration || 0;
    const frameCount = data.frame_count || 0;

    const origin = typeof window !== 'undefined' ? window.location.origin : '';

    // Build the OpenReel app URL with query params
    const appParams = new URLSearchParams({
      embedded: 'true',
      session_id: sessionId,
    });
    if (hasVideo && videoUrl) {
      appParams.set('video_url', origin + videoUrl);
    }
    if (hasAudio && audioUrl) {
      appParams.set('audio_url', origin + audioUrl);
    }
    if (fps) appParams.set('fps', fps.toString());

    // Pass ComfyUI theme colors so OpenReel can match the host UI.
    // The render() theme has basic tokens {bg, fg, border, accent}.
    // Also read --theme-* CSS vars injected by the iframe builder for extended colors.
    if (theme) {
      if (theme.bg) appParams.set('theme_bg', theme.bg);
      if (theme.fg) appParams.set('theme_fg', theme.fg);
      if (theme.border) appParams.set('theme_border', theme.border);
      if (theme.accent) appParams.set('theme_accent', theme.accent);

      // Try to read extended theme vars from the parent document's computed style
      if (typeof document !== 'undefined') {
        const rs = getComputedStyle(document.documentElement);
        const readVar = (name) => { const v = rs.getPropertyValue(name); return v ? v.trim() : ''; };
        const bgSecondary = readVar('--theme-bg-light') || readVar('--comfy-input-bg');
        const bgTertiary = readVar('--theme-bg-dark') || readVar('--comfy-menu-secondary-bg');
        const fgMuted = readVar('--theme-fg-muted') || readVar('--descrip-text');
        const inputBg = readVar('--theme-input-bg') || readVar('--comfy-input-bg');
        const scrollThumb = readVar('--theme-scrollbar-thumb') || readVar('--comfy-scrollbar-thumb');
        const scrollTrack = readVar('--theme-scrollbar-track') || readVar('--comfy-scrollbar-track');
        if (bgSecondary) appParams.set('theme_bg_secondary', bgSecondary);
        if (bgTertiary) appParams.set('theme_bg_tertiary', bgTertiary);
        if (fgMuted) appParams.set('theme_fg_muted', fgMuted);
        if (inputBg) appParams.set('theme_input_bg', inputBg);
        if (scrollThumb) appParams.set('theme_scrollbar_thumb', scrollThumb);
        if (scrollTrack) appParams.set('theme_scrollbar_track', scrollTrack);
      }
    }

    const appUrl = `${origin}/was/openreel_video/app/index.html?${appParams.toString()}`;

    // Info bar showing media metadata
    const contentType = data.type || 'openreel_video';
    const imageCount = data.image_count || 0;
    const infoItems = [];
    if (contentType === 'openreel_images') {
      infoItems.push(`${width}×${height}`);
      infoItems.push(`${imageCount} image${imageCount !== 1 ? 's' : ''}`);
    } else if (contentType === 'openreel_audio') {
      infoItems.push(`${data.channels || 1}ch`);
      infoItems.push(`${data.sample_rate || 44100}Hz`);
    } else if (contentType === 'openreel_combined') {
      infoItems.push(`${data.entry_count || 0} asset${(data.entry_count || 0) !== 1 ? 's' : ''}`);
    } else if (hasVideo) {
      infoItems.push(`${width}×${height}`);
      infoItems.push(`${frameCount} frames`);
      infoItems.push(`${fps} fps`);
    }
    if (duration > 0) {
      infoItems.push(`${duration.toFixed(2)}s`);
    }
    if (hasAudio && !hasVideo && contentType === 'openreel_video') {
      infoItems.push('Audio only');
    }

    const infoLabels = {
      'openreel_video': 'OpenReel Video',
      'openreel_images': 'OpenReel Images',
      'openreel_audio': 'OpenReel Audio',
      'openreel_combined': 'OpenReel Combined',
    };
    const infoLabel = infoLabels[contentType] || 'OpenReel';
    const infoText = infoItems.length > 0 ? infoItems.join(' · ') : 'No media';

    return `
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: ${theme?.bg || '#1a1a2e'};
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .openreel-container {
          display: flex;
          flex-direction: column;
          width: 100%;
          height: 100%;
        }
        .openreel-info-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 6px 12px;
          background: ${theme?.bg || '#1a1a2e'};
          border-bottom: 1px solid ${theme?.border || '#333'};
          color: ${theme?.fg || '#e0e0e0'};
          font-size: 12px;
          flex-shrink: 0;
        }
        .openreel-info-bar .info-left {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .openreel-info-bar .info-label {
          font-weight: 600;
          color: ${theme?.accent || '#7c5cbf'};
        }
        .openreel-info-bar .info-meta {
          opacity: 0.7;
        }
        .openreel-iframe-wrapper {
          flex: 1;
          position: relative;
          overflow: hidden;
        }
        .openreel-iframe {
          width: 100%;
          height: 100%;
          border: none;
        }
        .openreel-loading {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: ${theme?.bg || '#1a1a2e'};
          color: ${theme?.fg || '#e0e0e0'};
          z-index: 10;
        }
        .openreel-loading .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid ${theme?.border || '#333'};
          border-top-color: ${theme?.accent || '#7c5cbf'};
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin-bottom: 16px;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .openreel-loading .loading-text {
          font-size: 14px;
          opacity: 0.8;
        }
        .openreel-fallback {
          display: none;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          padding: 40px;
          text-align: center;
          color: ${theme?.fg || '#e0e0e0'};
        }
        .openreel-fallback .fallback-title {
          font-size: 18px;
          font-weight: 600;
          margin-bottom: 12px;
        }
        .openreel-fallback .fallback-message {
          font-size: 14px;
          opacity: 0.7;
          max-width: 400px;
          line-height: 1.5;
        }
      </style>

      <div class="openreel-container">
        <div class="openreel-info-bar">
          <div class="info-left">
            <span class="info-label">${infoLabel}</span>
            <span class="info-meta">${infoText}</span>
          </div>
        </div>

        <div class="openreel-iframe-wrapper">
          <div class="openreel-loading" id="openreel-loading">
            <div class="spinner"></div>
            <div class="loading-text">Loading OpenReel Video Editor...</div>
          </div>

          <div class="openreel-fallback" id="openreel-fallback">
            <div class="fallback-title">OpenReel Video Editor</div>
            <div class="fallback-message">
              The OpenReel app could not be loaded. Make sure the extension is properly installed
              with the built app files in web/openreel_app/.
            </div>
          </div>

          <iframe
            id="openreel-iframe"
            class="openreel-iframe"
            src="${appUrl}"
            allow="cross-origin-isolated"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"
            onload="document.getElementById('openreel-loading').style.display='none';"
            onerror="document.getElementById('openreel-loading').style.display='none'; document.getElementById('openreel-fallback').style.display='flex';"
          ></iframe>
        </div>
      </div>

      <script>
        // Listen for messages from the OpenReel iframe
        window.addEventListener('message', function(event) {
          if (!event.data || !event.data.type) return;

          if (event.data.type === 'openreel-video-output') {
            // Forward to parent (ComfyUI_Viewer's message handler)
            window.parent.postMessage(event.data, '*');
          }

          // Handle iframe load confirmation
          if (event.data.type === 'openreel-ready') {
            var loading = document.getElementById('openreel-loading');
            if (loading) loading.style.display = 'none';
          }
        });

        // Hide loading after timeout (fallback if onload doesn't fire)
        setTimeout(function() {
          var loading = document.getElementById('openreel-loading');
          if (loading && loading.style.display !== 'none') {
            loading.style.display = 'none';
          }
        }, 10000);
      </script>
    `;
  }

  static getStyles(theme) {
    return '';
  }
}

export default OpenReelVideoView;
