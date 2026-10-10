"use client";

import { useEffect, useState } from "react";
import { checkHealth, fetchModelInfo } from "./api";
import type { ModelInfo } from "./types";

const POLL_INTERVAL_MS = 5000;

// Pings GET /health every few seconds and loads GET /model-info once the API is up
export function useApiStatus() {
  const [online, setOnline] = useState<boolean | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      const ok = await checkHealth();
      if (cancelled) return;
      setOnline(ok);

      if (ok) {
        try {
          const info = await fetchModelInfo();
          if (!cancelled) setModelInfo(info);
        } catch {
          // Health passed but model-info failed; keep showing the last known info
        }
      }
    }

    const first = setTimeout(check, 0);
    const interval = setInterval(check, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearTimeout(first);
      clearInterval(interval);
    };
  }, []);

  return { online, modelInfo };
}
