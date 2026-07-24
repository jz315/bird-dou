import type { AiMode } from "../../types";

export type GuandanRank =
  | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10"
  | "jack" | "queen" | "king" | "ace";

export interface GuandanAction {
  index: number;
  kind: string;
  cards: number[];
  totalCards: number;
}

export interface GuandanEvent {
  sequence: number;
  actor: number;
  kind: string;
  cards: number[];
}

export interface GuandanTarget {
  actor: number;
  kind: string;
  cards: number[];
}

export interface GuandanResult {
  finishOrder: number[];
  winningTeam: "zero" | "one";
  levelAdvance: number;
  humanWon: boolean;
}

export interface GuandanState {
  schemaVersion: number;
  gameId: string;
  aiMode: string;
  phase: "card_play" | "terminal";
  humanSeat: number;
  humanTurn: boolean;
  currentPlayer: number | null;
  level: GuandanRank;
  hand: number[];
  cardsLeft: number[];
  target: GuandanTarget | null;
  legalActions: GuandanAction[];
  recentActions: GuandanEvent[];
  finishOrder: number[];
  result: GuandanResult | null;
  availableAiModes: AiMode[];
}

