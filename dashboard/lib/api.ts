import type { Features, ModelInfo, Prediction } from "./types";

// Set NEXT_PUBLIC_API_URL in .env.local to point at a different API
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5259";

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_URL}/model-info`);
  if (!res.ok) throw new Error(`GET /model-info failed with ${res.status}`);
  return res.json();
}

export async function predict(features: Features, timestamp: Date): Promise<Prediction> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ timestamp: toLocalIsoString(timestamp), features }),
  });
  if (!res.ok) throw new Error(`POST /predict failed with ${res.status}`);
  return res.json();
}

// The model's "hour" is the transaction's local hour. Date.toISOString() would send UTC
// ("...Z"), so build "2026-10-10T15:47:03+03:00" by hand to keep the local time and offset.
export function toLocalIsoString(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const abs = Math.abs(offset);

  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
    `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`
  );
}
