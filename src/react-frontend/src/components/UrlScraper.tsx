import { useState } from "react";
import { Globe, Loader2 } from "lucide-react";

interface Props {
  onScrape: (url: string) => Promise<void>;
}

export default function UrlScraper({ onScrape }: Props) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const handleScrape = async () => {
    if (!url.trim()) return;
    setLoading(true);
    try {
      await onScrape(url.trim());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-5 space-y-3">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
        <Globe size={14} />
        Scrape Images from URL
      </h2>

      <div className="flex gap-3">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleScrape()}
          placeholder="https://example.com/manga-chapter"
          className="flex-1 bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded-lg px-4 py-2.5 placeholder-gray-600 focus:ring-indigo-500 focus:border-indigo-500"
        />
        <button
          onClick={handleScrape}
          disabled={loading || !url.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
        >
          {loading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Globe size={16} />
          )}
          Scrape
        </button>
      </div>
    </div>
  );
}
