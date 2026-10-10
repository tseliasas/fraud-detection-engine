"use client";

import { useApiStatus } from "@/lib/useApiStatus";
import { useLiveFeed } from "@/lib/useLiveFeed";
import LiveFeed from "./LiveFeed";
import StatCards from "./StatCards";
import StatusBar from "./StatusBar";

export default function Dashboard() {
  const { online, modelInfo } = useApiStatus();
  const { feed, stats, paused, setPaused, speed, setSpeed, error, reset, total } = useLiveFeed();

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <StatusBar
        online={online}
        modelInfo={modelInfo}
        paused={paused}
        onTogglePause={() => setPaused(!paused)}
        speed={speed}
        onSpeedChange={setSpeed}
        onReset={reset}
      />

      {error && (
        <div className="rounded-md border border-red-900 bg-red-950/50 px-4 py-3 text-sm text-red-300">{error}</div>
      )}

      <StatCards stats={stats} />
      <LiveFeed feed={feed} />

      <footer className="text-xs text-zinc-500">
        Replays {total.toLocaleString()} unseen test-set transactions. Fraud is oversampled to ~5% so it appears
        often; the real rate is ~0.17%. Each transaction is timestamped now and scored live by the C# API.
      </footer>
    </main>
  );
}
