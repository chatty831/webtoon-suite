import { useCallback, useEffect, useState } from "react";
import {
  listColorizeModels,
  changeColorizeModel,
  changeColorizeParams,
} from "../api";
import { Settings, Loader2 } from "lucide-react";

export default function ModelConfig() {
  // Models
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(true);

  // Parameters
  const [denoiseSigma, setDenoiseSigma] = useState(50);
  const [imageTileSize, setImageTileSize] = useState(1280);

  // Status
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

  // Fetch models on mount
  useEffect(() => {
    listColorizeModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0) setSelectedModel(m[0]);
      })
      .catch(() => showToast("error", "Failed to load models"))
      .finally(() => setLoadingModels(false));
  }, [showToast]);

  // Model change
  const handleModelChange = useCallback(
    async (modelName: string) => {
      setSelectedModel(modelName);
      try {
        await changeColorizeModel(modelName);
        showToast("success", `Model changed to ${modelName}`);
      } catch {
        showToast("error", "Failed to change model");
      }
    },
    [showToast],
  );

  // Param change
  const handleParamChange = useCallback(async () => {
    try {
      await changeColorizeParams(denoiseSigma, imageTileSize);
      showToast("success", "Parameters updated");
    } catch {
      showToast("error", "Failed to update parameters");
    }
  }, [denoiseSigma, imageTileSize, showToast]);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-20 right-6 z-50 px-4 py-3 rounded-lg shadow-lg text-sm font-medium ${
            toast.type === "success"
              ? "bg-emerald-600 text-white"
              : "bg-red-600 text-white"
          }`}
        >
          {toast.msg}
        </div>
      )}

      <div className="flex items-center gap-3">
        <Settings size={24} className="text-indigo-400" />
        <h2 className="text-xl font-bold text-white">Model Configuration</h2>
      </div>

      {/* Colorization model selector */}
      <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6 space-y-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Colorization Model
        </h3>

        {loadingModels ? (
          <div className="flex items-center gap-2 text-gray-500">
            <Loader2 size={16} className="animate-spin" /> Loading models...
          </div>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded-lg px-4 py-2.5 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Parameters */}
      <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6 space-y-6">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Colorization Parameters
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {/* Denoise Sigma */}
          <div className="space-y-2">
            <label className="flex items-center justify-between text-sm text-gray-300">
              <span>Denoise Sigma</span>
              <span className="text-indigo-400 font-mono">{denoiseSigma}</span>
            </label>
            <input
              type="range"
              min={0}
              max={50}
              step={5}
              value={denoiseSigma}
              onChange={(e) => setDenoiseSigma(Number(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-gray-600">
              <span>0</span>
              <span>50</span>
            </div>
          </div>

          {/* Image Tile Size */}
          <div className="space-y-2">
            <label className="flex items-center justify-between text-sm text-gray-300">
              <span>Image Tile Size</span>
              <span className="text-indigo-400 font-mono">{imageTileSize}</span>
            </label>
            <input
              type="range"
              min={576}
              max={1920}
              step={32}
              value={imageTileSize}
              onChange={(e) => setImageTileSize(Number(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-gray-600">
              <span>576</span>
              <span>1920</span>
            </div>
          </div>
        </div>

        <button
          onClick={handleParamChange}
          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Apply Parameters
        </button>
      </div>
    </div>
  );
}
