import type { ModelInfo } from "@/lib/types";
import { SPEEDS } from "@/lib/useLiveFeed";

type Props = {
  online: boolean | null;
  modelInfo: ModelInfo | null;
  paused: boolean;
  onTogglePause: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  onReset: () => void;
};

export default function StatusBar({ online, modelInfo, paused, onTogglePause, speed, onSpeedChange, onReset }: Props) {
  const dotColor = online === null ? "bg-zinc-500" : online ? "bg-emerald-400" : "bg-red-500";
  const statusText = online === null ? "Connecting…" : online ? "API online" : "API offline";

  return (
    <header className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 pb-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">🛡 Fraud Detection Dashboard</h1>
        <div className="mt-1 flex items-center gap-2 text-sm text-zinc-400">
          <span className={`inline-block h-2 w-2 rounded-full ${dotColor}`} />
          <span>{statusText}</span>
          {modelInfo && (
            <span>
              · model {modelInfo.version} · threshold {modelInfo.threshold.toFixed(3)}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onTogglePause}
          className="rounded-md bg-zinc-800 px-3 py-1.5 text-sm hover:bg-zinc-700"
        >
          {paused ? "▶ Resume" : "⏸ Pause"}
        </button>

        <label className="flex items-center gap-1 text-sm text-zinc-400">
          Speed
          <select
            value={speed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="rounded-md bg-zinc-800 px-2 py-1.5 text-zinc-100"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}x
              </option>
            ))}
          </select>
        </label>

        <button onClick={onReset} className="rounded-md bg-zinc-800 px-3 py-1.5 text-sm hover:bg-zinc-700">
          ↺ Reset
        </button>
      </div>
    </header>
  );
}
