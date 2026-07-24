import type { GuandanRank } from "./types";

// 🔴 修复：明确指定花色，确保红桃在第一位，红/黑在前两位
const SUITS = ["♥", "♦", "♠", "♣"] as const; 
const RANK_KEYS: GuandanRank[] = [
  "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace",
];
const RANK_LABELS: Record<GuandanRank, string> = {
  "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
  "8": "8", "9": "9", "10": "10", jack: "J", queen: "Q", king: "K", ace: "A",
};

export interface PhysicalCardView {
  id: number;
  rankIndex: number;
  label: string;
  suit: string;
  red: boolean;
  joker: boolean;
  wild: boolean;
  strength: number;
}

export function cardView(id: number, level: GuandanRank): PhysicalCardView {
  const face = id % 54;
  const rankIndex = face < 52 ? Math.floor(face / 4) : face === 52 ? 13 : 14;
  const joker = rankIndex >= 13;
  const suit = joker ? "JOKER" : SUITS[face % 4];
  const levelIndex = RANK_KEYS.indexOf(level);
  
  // 🔴 修复：显式指定红桃为逢人配，红心/方块为红色
  const wild = !joker && rankIndex === levelIndex && suit === "♥"; 
  const red = joker ? rankIndex === 14 : suit === "♥" || suit === "♦";
  
  const strength = joker ? rankIndex + 1 : rankIndex === levelIndex ? 13 : rankIndex;
  return {
    id, rankIndex,
    label: joker ? (rankIndex === 13 ? "小王" : "大王") : RANK_LABELS[RANK_KEYS[rankIndex]],
    suit, red, joker, wild, strength,
  };
}

export function sortedCards(cards: number[], level: GuandanRank): PhysicalCardView[] {
  return cards
    .map((id) => cardView(id, level))
    .sort((left, right) =>
      left.strength - right.strength
      || left.rankIndex - right.rankIndex
      || left.suit.localeCompare(right.suit)
      || left.id - right.id,
    );
}

export const MOVE_LABELS: Record<string, string> = {
  pass: "不出", single: "单张", pair: "对子", triple: "三张",
  full_house: "三带二", straight: "顺子", pair_straight: "三连对",
  triple_straight: "钢板", bomb: "炸弹", straight_flush: "同花顺", four_jokers: "四王",
};

export function levelLabel(level: GuandanRank): string {
  return RANK_LABELS[level];
}