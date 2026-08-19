import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";

const ACCEPTED = ".jpg,.jpeg,.png,.webp,.bmp";

function naturalSort(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // strip "data:...;base64," prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

interface Props {
  onFilesSelected: (base64Images: string[]) => void;
}

export default function FileUploader({ onFilesSelected }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [fileNames, setFileNames] = useState<string[]>([]);

  const processFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files).filter((f) =>
        /\.(jpe?g|png|webp|bmp)$/i.test(f.name),
      );
      if (arr.length === 0) return;
      arr.sort((a, b) => naturalSort(a.name, b.name));
      setFileNames(arr.map((f) => f.name));
      const b64 = await Promise.all(arr.map(fileToBase64));
      onFilesSelected(b64);
    },
    [onFilesSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
    },
    [processFiles],
  );

  const clearFiles = useCallback(() => {
    setFileNames([]);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
        dragging
          ? "border-indigo-500 bg-indigo-500/10"
          : "border-gray-700 hover:border-gray-500 bg-gray-900/50"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        multiple
        className="hidden"
        onChange={(e) => e.target.files && processFiles(e.target.files)}
      />
      <Upload className="mx-auto mb-3 text-gray-500" size={32} />
      <p className="text-gray-400 text-sm">
        Drag &amp; drop images here or click to browse
      </p>
      <p className="text-gray-600 text-xs mt-1">
        JPG, JPEG, PNG, WebP, BMP
      </p>

      {fileNames.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2 justify-center">
          {fileNames.map((name) => (
            <span
              key={name}
              className="inline-flex items-center gap-1 rounded-full bg-gray-800 px-3 py-1 text-xs text-gray-300"
            >
              {name}
            </span>
          ))}
          <button
            onClick={(e) => {
              e.stopPropagation();
              clearFiles();
            }}
            className="inline-flex items-center gap-1 rounded-full bg-red-900/50 px-2 py-1 text-xs text-red-300 hover:bg-red-900"
          >
            <X size={12} /> Clear
          </button>
        </div>
      )}
    </div>
  );
}
