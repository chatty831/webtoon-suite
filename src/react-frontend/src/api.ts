const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:9374";

// ---------------------------------------------------------------------------
// Liveness
// ---------------------------------------------------------------------------

export async function livenessProbe(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/liveness-probe`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

// ---------------------------------------------------------------------------
// Scrape images from URL
// ---------------------------------------------------------------------------

export async function scrapeImages(
  url: string,
  sessionId?: string,
): Promise<string[]> {
  const params: Record<string, string> = { url };
  if (sessionId) params.session_id = sessionId;
  const res = await fetch(
    `${API_BASE}/scrape-images?${new URLSearchParams(params)}`,
  );
  if (!res.ok) throw new Error(`Scrape failed: ${res.statusText}`);
  const data = await res.json();
  return data.scraped_images;
}

// ---------------------------------------------------------------------------
// Delete cache
// ---------------------------------------------------------------------------

export async function deleteCache(images: (string | null)[]): Promise<void> {
  const res = await fetch(`${API_BASE}/delete-cache`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  });
  if (!res.ok) throw new Error(`Delete cache failed: ${res.statusText}`);
}

// ---------------------------------------------------------------------------
// Process images (batch)
// ---------------------------------------------------------------------------

export interface ProcessParams {
  translate: boolean;
  colorize: boolean;
  upscale: boolean;
  colorize_method?: string;
  comfyui_url?: string;
}

export async function processImages(
  images: (string | null)[],
  params: ProcessParams,
): Promise<string[]> {
  const qs = new URLSearchParams({
    translate: String(params.translate),
    colorize: String(params.colorize),
    upscale: String(params.upscale),
    colorize_method: params.colorize_method ?? "gan",
    comfyui_url: params.comfyui_url ?? "http://localhost:9010",
  });
  const res = await fetch(`${API_BASE}/process-images?${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  });
  if (!res.ok) throw new Error(`Process failed: ${res.statusText}`);
  const data = await res.json();
  return data.translated_images;
}

// ---------------------------------------------------------------------------
// Process images (streaming NDJSON)
// ---------------------------------------------------------------------------

export interface StreamChunk {
  index: number;
  image: string | null;
  error?: string;
}

export async function* processImagesStream(
  images: (string | null)[],
  params: ProcessParams,
  signal?: AbortSignal,
  sessionId?: string,
): AsyncGenerator<StreamChunk> {
  const qs = new URLSearchParams({
    translate: String(params.translate),
    colorize: String(params.colorize),
    upscale: String(params.upscale),
    colorize_method: params.colorize_method ?? "gan",
    comfyui_url: params.comfyui_url ?? "http://localhost:9010",
  });
  if (sessionId) qs.set("session_id", sessionId);
  const res = await fetch(`${API_BASE}/process-images-stream?${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
    signal,
  });
  if (!res.ok) throw new Error(`Stream failed: ${res.statusText}`);
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) {
        yield JSON.parse(line) as StreamChunk;
      }
    }
  }
  if (buffer.trim()) {
    yield JSON.parse(buffer) as StreamChunk;
  }
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

export async function cancelSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/session/${sessionId}/cancel`, { method: "POST" });
}

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

export async function listColorizeModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/list-colorize-models`);
  if (!res.ok) throw new Error("Failed to list models");
  const data = await res.json();
  return data.models;
}

export async function changeColorizeModel(modelName: string): Promise<void> {
  const fullPath = `models/colorize_models/${modelName}`;
  const res = await fetch(
    `${API_BASE}/colorize-model-change?${new URLSearchParams({ model_path: fullPath })}`,
  );
  if (!res.ok) throw new Error(`Model change failed: ${res.statusText}`);
}

export async function changeColorizeParams(
  denoiseSigma: number,
  imageTileSize: number,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/colorize-param-change?${new URLSearchParams({
      denoise_sigma: String(denoiseSigma),
      image_tile_size: String(imageTileSize),
    })}`,
  );
  if (!res.ok) throw new Error(`Param change failed: ${res.statusText}`);
}
