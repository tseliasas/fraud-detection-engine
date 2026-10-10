import type { Stats } from "@/lib/types";

function percent(numerator: number, denominator: number): string {
  return denominator === 0 ? "—" : `${((numerator / denominator) * 100).toFixed(1)}%`;
}

type CardProps = { label: string; value: number; note?: string; accent: string };

function Card({ label, value, note, accent }: CardProps) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="text-xs uppercase tracking-wide text-zinc-400">{label}</div>
      <div className={`mt-1 text-3xl font-semibold tabular-nums ${accent}`}>{value.toLocaleString()}</div>
      {note && <div className="mt-1 text-xs text-zinc-500">{note}</div>}
    </div>
  );
}

export default function StatCards({ stats }: { stats: Stats }) {
  // Same definitions as Phase 1: recall = caught / all frauds, precision = caught / all flags
  const recall = percent(stats.caught, stats.caught + stats.missed);
  const precision = percent(stats.caught, stats.flagged);

  return (
    <section className="grid grid-cols-2 gap-3 md:grid-cols-5">
      <Card label="Processed" value={stats.processed} accent="text-zinc-100" />
      <Card label="Flagged" value={stats.flagged} accent="text-red-400" />
      <Card label="Fraud caught" value={stats.caught} note={`recall ${recall}`} accent="text-emerald-400" />
      <Card label="Fraud missed" value={stats.missed} accent="text-amber-400" />
      <Card label="False alarms" value={stats.falseAlarms} note={`precision ${precision}`} accent="text-orange-400" />
    </section>
  );
}
