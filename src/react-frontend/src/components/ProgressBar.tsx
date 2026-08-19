interface Props {
  current: number;
  total: number;
}

export default function ProgressBar({ current, total }: Props) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>
          Processed {current} / {total} images
        </span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
