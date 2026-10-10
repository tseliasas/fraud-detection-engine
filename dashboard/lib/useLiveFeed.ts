"use client";

import { useEffect, useRef, useState } from "react";
import { API_URL, predict } from "./api";
import type { DemoTransaction, FeedItem, Outcome, Stats } from "./types";

export const SPEEDS = [1, 2, 5, 10];
const BASE_INTERVAL_MS = 1000;
const RETRY_INTERVAL_MS = 3000;
const MAX_FEED_ITEMS = 50;

const EMPTY_STATS: Stats = { processed: 0, flagged: 0, caught: 0, missed: 0, falseAlarms: 0 };

function classify(flagged: boolean, actualFraud: boolean): Outcome {
  if (flagged && actualFraud) return "caught";
  if (!flagged && actualFraud) return "missed";
  if (flagged && !actualFraud) return "falseAlarm";
  return "clear";
}

function addToStats(stats: Stats, outcome: Outcome): Stats {
  return {
    processed: stats.processed + 1,
    flagged: stats.flagged + (outcome === "caught" || outcome === "falseAlarm" ? 1 : 0),
    caught: stats.caught + (outcome === "caught" ? 1 : 0),
    missed: stats.missed + (outcome === "missed" ? 1 : 0),
    falseAlarms: stats.falseAlarms + (outcome === "falseAlarm" ? 1 : 0),
  };
}

// Replays demo transactions one at a time through POST /predict and keeps the running results
export function useLiveFeed() {
  const [transactions, setTransactions] = useState<DemoTransaction[]>([]);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [stats, setStats] = useState<Stats>(EMPTY_STATS);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState<string | null>(null);

  // How far through the demo file we are; a ref so changing it doesn't re-render
  const nextIndex = useRef(0);

  useEffect(() => {
    fetch("/demo_transactions.json")
      .then((res) => res.json())
      .then(setTransactions)
      .catch(() => setError("Could not load demo_transactions.json"));
  }, []);

  useEffect(() => {
    if (paused || transactions.length === 0) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    // Wait for each prediction before scheduling the next, so requests never pile up
    async function step() {
      const seq = nextIndex.current;
      const tx = transactions[seq % transactions.length];
      const timestamp = new Date();
      let delay = BASE_INTERVAL_MS / speed;

      try {
        const prediction = await predict(tx.features, timestamp);
        if (cancelled) return;

        nextIndex.current = seq + 1;
        const outcome = classify(prediction.isFraud, tx.actualFraud);
        const item: FeedItem = {
          seq,
          timestamp,
          amount: tx.features.Amount,
          prediction,
          actualFraud: tx.actualFraud,
          outcome,
        };

        setFeed((prev) => [item, ...prev].slice(0, MAX_FEED_ITEMS));
        setStats((prev) => addToStats(prev, outcome));
        setError(null);
      } catch {
        if (cancelled) return;
        setError(`Can't reach the API at ${API_URL}. Is it running?`);
        delay = RETRY_INTERVAL_MS;
      }

      timer = setTimeout(step, delay);
    }

    timer = setTimeout(step, 0);

    // Runs when paused, speed changes or the page closes: stop the loop
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [paused, speed, transactions]);

  function reset() {
    nextIndex.current = 0;
    setFeed([]);
    setStats(EMPTY_STATS);
  }

  return { feed, stats, paused, setPaused, speed, setSpeed, error, reset, total: transactions.length };
}
