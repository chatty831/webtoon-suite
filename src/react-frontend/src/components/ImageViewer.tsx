import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, List, LayoutGrid } from "lucide-react";

interface Props {
  images: (string | null)[];
}

export default function ImageViewer({ images }: Props) {
  const [mode, setMode] = useState<"list" | "paginated">("paginated");
  const [page, setPage] = useState(0);

  // Reset page only when a new batch is loaded (length changes)
  useEffect(() => setPage(0), [images.length]);

  // Clamp page
  const totalPages = images.length;
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));

  const goNext = useCallback(
    () => setPage((p) => Math.min(p + 1, totalPages - 1)),
    [totalPages],
  );
  const goPrev = useCallback(() => setPage((p) => Math.max(p - 1, 0)), []);

  // Keyboard navigation
  useEffect(() => {
    if (mode !== "paginated") return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") goPrev();
      if (e.key === "ArrowRight") goNext();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [mode, goNext, goPrev]);

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Image Preview
        </h2>
        <div className="flex rounded-lg overflow-hidden border border-gray-700">
          <button
            onClick={() => setMode("paginated")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === "paginated"
                ? "bg-indigo-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            <LayoutGrid size={14} /> Paginated
          </button>
          <button
            onClick={() => setMode("list")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === "list"
                ? "bg-indigo-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            <List size={14} /> List
          </button>
        </div>
      </div>

      {mode === "list" ? (
        <div className="flex flex-col items-center">
          {images.map((img, i) => (
            <ImageCard key={i} image={img} index={i} seamless />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Page indicator */}
          <p className="text-center text-sm text-gray-500">
            Page {currentPage + 1} of {totalPages}
          </p>

          {/* Image */}
          <ImageCard image={images[currentPage]} index={currentPage} />

          {/* Navigation */}
          {totalPages > 1 && (
            <div className="flex justify-between">
              <button
                onClick={goPrev}
                disabled={currentPage <= 0}
                className="flex items-center gap-1 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed text-gray-300 rounded-lg text-sm transition-colors"
              >
                <ChevronLeft size={16} /> Previous
              </button>
              <button
                onClick={goNext}
                disabled={currentPage >= totalPages - 1}
                className="flex items-center gap-1 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed text-gray-300 rounded-lg text-sm transition-colors"
              >
                Next <ChevronRight size={16} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ImageCard({
  image,
  index,
  seamless = false,
}: {
  image: string | null;
  index: number;
  seamless?: boolean;
}) {
  if (!image) {
    return (
      <div className="flex justify-center w-full">
        <div className="w-[750px] max-w-full aspect-square border-2 border-dashed border-gray-700 rounded-xl flex items-center justify-center text-gray-600 text-lg">
          Image {index + 1} — processing...
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-center w-full">
      <img
        src={`data:image/jpeg;base64,${image}`}
        alt={`Image ${index + 1}`}
        className={`max-w-[750px] w-full ${seamless ? "block" : "rounded-lg shadow-lg"}`}
      />
    </div>
  );
}
