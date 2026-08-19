import { Languages, Palette, ArrowUpFromLine } from "lucide-react";

interface Props {
  translate: boolean;
  colorize: boolean;
  upscale: boolean;
  colorizeMethod: "GAN" | "ComfyUI";
  comfyuiUrl: string;
  onTranslateChange: (v: boolean) => void;
  onColorizeChange: (v: boolean) => void;
  onUpscaleChange: (v: boolean) => void;
  onColorizeMethodChange: (v: "GAN" | "ComfyUI") => void;
  onComfyuiUrlChange: (v: string) => void;
}

function Toggle({
  label,
  icon: Icon,
  checked,
  onChange,
}: {
  label: string;
  icon: React.ElementType;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer select-none">
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="w-9 h-5 bg-gray-700 rounded-full peer-checked:bg-indigo-600 transition-colors" />
        <div className="absolute left-0.5 top-0.5 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-4" />
      </div>
      <Icon size={16} className="text-gray-400" />
      <span className="text-sm text-gray-300">{label}</span>
    </label>
  );
}

export default function ControlPanel({
  translate,
  colorize,
  upscale,
  colorizeMethod,
  comfyuiUrl,
  onTranslateChange,
  onColorizeChange,
  onUpscaleChange,
  onColorizeMethodChange,
  onComfyuiUrlChange,
}: Props) {
  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-5 space-y-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
        Processing Options
      </h2>

      <div className="flex flex-wrap gap-6">
        <Toggle
          label="Translate"
          icon={Languages}
          checked={translate}
          onChange={onTranslateChange}
        />
        <Toggle
          label="Colorize"
          icon={Palette}
          checked={colorize}
          onChange={onColorizeChange}
        />
        <Toggle
          label="Upscale"
          icon={ArrowUpFromLine}
          checked={upscale}
          onChange={onUpscaleChange}
        />
      </div>

      {colorize && (
        <div className="flex flex-wrap gap-4 items-end pt-2 border-t border-gray-800">
          <div>
            <label className="block text-xs text-gray-500 mb-1">
              Colorize Method
            </label>
            <select
              value={colorizeMethod}
              onChange={(e) =>
                onColorizeMethodChange(e.target.value as "GAN" | "ComfyUI")
              }
              className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="GAN">GAN</option>
              <option value="ComfyUI">ComfyUI</option>
            </select>
          </div>

          {colorizeMethod === "ComfyUI" && (
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-gray-500 mb-1">
                ComfyUI URL
              </label>
              <input
                type="text"
                value={comfyuiUrl}
                onChange={(e) => onComfyuiUrlChange(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
