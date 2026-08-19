import { Play, Download, Trash2, Loader2 } from "lucide-react";

interface Props {
  hasOriginal: boolean;
  hasTranslated: boolean;
  processing: boolean;
  onProcess: () => void;
  onDownload: () => void;
  onDeleteCache: () => void;
}

export default function ActionButtons({
  hasOriginal,
  hasTranslated,
  processing,
  onProcess,
  onDownload,
  onDeleteCache,
}: Props) {
  return (
    <div className="flex flex-wrap gap-3">
      <button
        onClick={onProcess}
        disabled={!hasOriginal || processing}
        className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
      >
        {processing ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          <Play size={16} />
        )}
        {processing ? "Processing..." : "Process"}
      </button>

      <button
        onClick={onDownload}
        disabled={!hasTranslated}
        className="flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
      >
        <Download size={16} />
        Download ZIP
      </button>

      <button
        onClick={onDeleteCache}
        disabled={!hasOriginal}
        className="flex items-center gap-2 px-6 py-2.5 bg-red-700 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
      >
        <Trash2 size={16} />
        Delete Cache
      </button>
    </div>
  );
}
