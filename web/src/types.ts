export type Phase = "bidding" | "doubling" | "card_play" | "terminal";

export interface AiMode {
  id: string;
  label: string;
  description: string;
  recommended: boolean;
}

export interface GameAction {
  index: number;
  phase: Phase;
  kind: string;
  label: string;
  cards: number[];
  totalCards: number;
  actor?: number;
  sequence?: number;
}

export interface GameResult {
  allPass: boolean;
  humanWon: boolean | null;
  winnerSeat: number | null;
  rawPayoff: number[];
  message: string;
}

export interface GameState {
  schemaVersion: number;
  gameId: string;
  seed: number;
  humanSeat: number;
  aiMode: string;
  phase: Phase;
  humanTurn: boolean;
  terminal: boolean;
  currentPlayer: number | null;
  landlord: number | null;
  role: "unassigned" | "landlord" | "farmer";
  hand: number[];
  cardsLeft: number[];
  bottomCards: number[];
  multiplier: number;
  bombCount: number;
  lastNonPass: GameAction | null;
  legalActions: GameAction[];
  recentActions: GameAction[];
  result: GameResult | null;
  availableAiModes: AiMode[];
}

export interface ApiError {
  error?: string;
}
