// Shapes of the data moving through the dashboard (TypeScript's version of Contracts.cs)

export type Features = Record<string, number>;

// One row of public/demo_transactions.json
export type DemoTransaction = {
  id: number;
  actualFraud: boolean;
  features: Features;
};

// What POST /predict returns
export type Prediction = {
  fraudProbability: number;
  isFraud: boolean;
  threshold: number;
  modelVersion: string;
};

// What GET /model-info returns
export type ModelInfo = {
  modelFile: string;
  inputName: string;
  version: string;
  features: string[];
  threshold: number;
};

// Model's call vs the truth, the four cells of the confusion matrix
export type Outcome = "caught" | "missed" | "falseAlarm" | "clear";

export type FeedItem = {
  seq: number;
  timestamp: Date;
  amount: number;
  prediction: Prediction;
  actualFraud: boolean;
  outcome: Outcome;
};

export type Stats = {
  processed: number;
  flagged: number;
  caught: number;
  missed: number;
  falseAlarms: number;
};
