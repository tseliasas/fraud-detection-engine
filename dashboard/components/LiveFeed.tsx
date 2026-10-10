import type { FeedItem, Outcome } from "@/lib/types";

const OUTCOME_LABELS: Record<Outcome, { text: string; className: string }> = {
  caught: { text: "✓ fraud caught", className: "text-emerald-400" },
  missed: { text: "✗ fraud missed", className: "text-amber-400" },
  falseAlarm: { text: "⚠ false alarm", className: "text-orange-400" },
  clear: { text: "legit", className: "text-zinc-500" },
};

function formatProbability(p: number): string {
  return p < 0.0001 ? "<0.01%" : `${(p * 100).toFixed(2)}%`;
}

function ProbabilityBar({ probability, threshold, flagged }: { probability: number; threshold: number; flagged: boolean }) {
  return (
    <div className="relative h-2 w-32 rounded-full bg-zinc-800">
      <div
        className={`h-2 rounded-full ${flagged ? "bg-red-500" : "bg-emerald-500"}`}
        style={{ width: `${Math.max(probability * 100, 1)}%` }}
      />
      {/* Thin marker showing where the threshold sits on the bar */}
      <div className="absolute top-[-2px] h-3 w-px bg-zinc-300" style={{ left: `${threshold * 100}%` }} />
    </div>
  );
}

export default function LiveFeed({ feed }: { feed: FeedItem[] }) {
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900">
      <h2 className="border-b border-zinc-800 px-4 py-3 text-sm font-medium text-zinc-300">
        Live feed <span className="text-zinc-500">(newest first)</span>
      </h2>

      {feed.length === 0 ? (
        <p className="px-4 py-8 text-center text-sm text-zinc-500">Waiting for transactions…</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-zinc-500">
              <tr>
                <th className="px-4 py-2 font-normal">Time</th>
                <th className="px-4 py-2 text-right font-normal">Amount</th>
                <th className="px-4 py-2 font-normal">Fraud probability</th>
                <th className="px-4 py-2 font-normal">Decision</th>
                <th className="px-4 py-2 font-normal">Actual</th>
              </tr>
            </thead>
            <tbody>
              {feed.map((item) => {
                const flagged = item.prediction.isFraud;
                const outcome = OUTCOME_LABELS[item.outcome];

                return (
                  <tr key={item.seq} className={`border-t border-zinc-800 ${flagged ? "bg-red-950/30" : ""}`}>
                    <td className="px-4 py-2 font-mono text-zinc-400">{item.timestamp.toLocaleTimeString()}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">${item.amount.toFixed(2)}</td>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-3">
                        <ProbabilityBar
                          probability={item.prediction.fraudProbability}
                          threshold={item.prediction.threshold}
                          flagged={flagged}
                        />
                        <span className="w-16 font-mono tabular-nums text-zinc-300">
                          {formatProbability(item.prediction.fraudProbability)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-2">
                      {flagged ? (
                        <span className="rounded bg-red-500/20 px-2 py-0.5 text-xs font-semibold text-red-400">
                          🚨 FLAGGED
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-500">ok</span>
                      )}
                    </td>
                    <td className={`px-4 py-2 text-xs ${outcome.className}`}>{outcome.text}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
