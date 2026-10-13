import { useCallback, useRef, useState } from "react";
import FileUploader from "../components/FileUploader";
import ControlPanel from "../components/ControlPanel";
import UrlScraper from "../components/UrlScraper";
import ActionButtons from "../components/ActionButtons";
import ImageViewer from "../components/ImageViewer";
import ProgressBar from "../components/ProgressBar";
import {
  processImagesStream,
  scrapeImages,
  cancelSession,
  deleteCache,
  type ProcessParams,
} from "../api";

export default function ImageProcessing() {
  // Unique session ID for this tab (generated once on mount)
  const [sessionId] = useState(() => crypto.randomUUID());

  // Images state
  const [originalImages, setOriginalImages] = useState<(string | null)[]>([]);
  const [translatedImages, setTranslatedImages] = useState<(string | null)[]>(
    [],
  );

  // Processing flags
  const [translate, setTranslate] = useState(true);
  const [colorize, setColorize] = useState(true);
  const [upscale, setUpscale] = useState(true);
  const [colorizeMethod, setColorizeMethod] = useState<"GAN" | "ComfyUI">(
    "GAN",
  );
  const [comfyuiUrl, setComfyuiUrl] = useState("http://localhost:9010");

  // Progress
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const abortRef = useRef<AbortController | null>(null);

  // Notification
  const [toast, setToast] = useState<{
    type: "success" | "error";
    msg: string;
  } | null>(null);

  const showToast = useCallback(
    (type: "success" | "error", msg: string) => {
      setToast({ type, msg });
      setTimeout(() => setToast(null), 4000);
    },
    [],
  );

  // ---- File upload ----
  const handleFilesSelected = useCallback(
    (base64Images: string[]) => {
      setTranslatedImages([]);
      setOriginalImages(base64Images);
      showToast("success", `Loaded ${base64Images.length} image(s)`);
    },
    [showToast],
  );

  // ---- URL scraping ----
  const handleScrape = useCallback(
    async (url: string) => {
      try {
        const imgs = await scrapeImages(url, sessionId);
        setTranslatedImages([]);
        setOriginalImages(imgs);
        showToast("success", `Scraped ${imgs.length} image(s) from URL`);
      } catch (err: unknown) {
        showToast(
          "error",
          err instanceof Error ? err.message : "Scraping failed",
        );
      }
    },
    [showToast, sessionId],
  );

  // ---- Process images (streaming) ----
  const handleProcess = useCallback(async () => {
    if (originalImages.length === 0) return;
    setProcessing(true);
    setProgress({ current: 0, total: originalImages.length });

    const results: (string | null)[] = new Array(originalImages.length).fill(
      null,
    );
    setTranslatedImages([...results]);

    const ac = new AbortController();
    abortRef.current = ac;

    const params: ProcessParams = {
      translate,
      colorize,
      upscale,
      colorize_method: colorizeMethod.toLowerCase(),
      comfyui_url: comfyuiUrl,
    };

    try {
      let completed = 0;
      for await (const chunk of processImagesStream(
        originalImages,
        params,
        ac.signal,
        sessionId,
      )) {
        if (chunk.index != null && chunk.image) {
          results[chunk.index] = chunk.image;
        }
        completed++;
        setProgress({ current: completed, total: originalImages.length });
        setTranslatedImages([...results]);
      }
      setOriginalImages([]);
      showToast("success", "Processing complete!");
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") {
        showToast(
          "error",
          err instanceof Error ? err.message : "Processing failed",
        );
      }
    } finally {
      setProcessing(false);
      abortRef.current = null;
    }
  }, [
    originalImages,
    translate,
    colorize,
    upscale,
    colorizeMethod,
    comfyuiUrl,
    showToast,
    sessionId,
  ]);

  // ---- Cancel processing ----
  const handleCancel = useCallback(() => {
    // Tell server to drain all queued tasks for this session
    cancelSession(sessionId).catch(() => {});
    abortRef.current?.abort();
    abortRef.current = null;
    setProcessing(false);
    showToast("error", "Processing cancelled");
  }, [showToast, sessionId]);

  // ---- Delete cache ----
  const handleDeleteCache = useCallback(async () => {
    if (originalImages.length === 0) return;
    try {
      await deleteCache(originalImages);
      showToast("success", "Cache deleted");
    } catch {
      showToast("error", "Failed to delete cache");
    }
  }, [originalImages, showToast]);

  // ---- Download ZIP ----
  const handleDownload = useCallback(async () => {
    const images = translatedImages.filter(Boolean) as string[];
    if (images.length === 0) return;

    const JSZip = (await import("jszip")).default;
    const { saveAs } = await import("file-saver");
    const zip = new JSZip();
    images.forEach((b64, i) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      zip.file(`translated_image_${i + 1}.jpg`, bytes);
    });
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, "translated_images.zip");
  }, [translatedImages]);

  // Determine which images to display
  const displayImages =
    translatedImages.length > 0 ? translatedImages : originalImages;

  return (
    <div className="space-y-6">
      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-20 right-6 z-50 px-4 py-3 rounded-lg shadow-lg text-sm font-medium transition-all ${
            toast.type === "success"
              ? "bg-emerald-600 text-white"
              : "bg-red-600 text-white"
          }`}
        >
          {toast.msg}
        </div>
      )}

      {/* File uploader */}
      <FileUploader onFilesSelected={handleFilesSelected} />

      {/* Control panel */}
      <ControlPanel
        translate={translate}
        colorize={colorize}
        upscale={upscale}
        colorizeMethod={colorizeMethod}
        comfyuiUrl={comfyuiUrl}
        onTranslateChange={setTranslate}
        onColorizeChange={setColorize}
        onUpscaleChange={setUpscale}
        onColorizeMethodChange={setColorizeMethod}
        onComfyuiUrlChange={setComfyuiUrl}
      />

      {/* URL Scraper */}
      <UrlScraper onScrape={handleScrape} />

      {/* Action buttons */}
      <ActionButtons
        hasOriginal={originalImages.length > 0}
        hasTranslated={translatedImages.filter(Boolean).length > 0}
        processing={processing}
        onProcess={handleProcess}
        onCancel={handleCancel}
        onDownload={handleDownload}
        onDeleteCache={handleDeleteCache}
      />

      {/* Progress bar */}
      {processing && (
        <ProgressBar current={progress.current} total={progress.total} />
      )}

      {/* Image viewer */}
      {displayImages.length > 0 && <ImageViewer images={displayImages} />}
    </div>
  );
}
