import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createGuandanGame, playGuandanCards } from "./api";
import type { GuandanAction, GuandanState } from "./types";

function sameCards(left: number[], right: number[]) {
  if (left.length !== right.length) return false;
  const a = [...left].sort((x, y) => x - y);
  const b = [...right].sort((x, y) => x - y);
  return a.every((value, index) => value === b[index]);
}

export function useGuandanGame() {
  const [game, setGame] = useState<GuandanState | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [levelChoice, setLevelChoice] = useState(0);
  const [aiMode, setAiMode] = useState("heuristic");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showRules, setShowRules] = useState(false);
  const [hintIndex, setHintIndex] = useState(0);

  const startGame = useCallback(async (level = levelChoice) => {
    setBusy(true); setError(null); setSelected(new Set());
    try {
      const state = await createGuandanGame(level, aiMode);
      setGame(state); setAiMode(state.aiMode); setHintIndex(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "掼蛋开局失败");
    } finally { setBusy(false); }
  }, [aiMode, levelChoice]);

  // 🔴 修复：使用 ref 解决 useEffect 依赖问题，避免过期闭包
  const startGameRef = useRef(startGame);
  startGameRef.current = startGame;
  useEffect(() => { void startGameRef.current(0); }, []); 

  const suggestions = useMemo(() => game?.legalActions.filter((action) => action.kind !== "pass") ?? [], [game?.legalActions]);
  const passAction = useMemo(() => game?.legalActions.find((action) => action.kind === "pass"), [game?.legalActions]);
  const selectedCards = useMemo(() => [...selected], [selected]);
  const selectedAction = suggestions.find((action) => sameCards(action.cards, selectedCards));

  const submit = useCallback(async (cards: number[] | null) => {
    if (!game || busy) return;
    setBusy(true); setError(null);
    try {
      setGame(await playGuandanCards(game.gameId, cards));
      setSelected(new Set()); setHintIndex(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "动作提交失败");
    } finally { setBusy(false); }
  }, [busy, game]);

  const toggleCard = (id: number) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectSuggestion = (action: GuandanAction) => setSelected(new Set(action.cards));
  const showHint = () => {
    if (!suggestions.length) return;
    selectSuggestion(suggestions[hintIndex % suggestions.length]);
    setHintIndex((value) => value + 1);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // 🔴 修复：防止在输入框/选择器中误触游戏快捷键
      if ((event.target as HTMLElement).closest("input, select, textarea")) return;
      
      if (event.key === "Escape") setSelected(new Set());
      
      // 🔴 修复：必须选了牌且牌型合法才能按回车出牌
      if (event.key === "Enter" && selectedCards.length && selectedAction) {
        void submit(selectedCards);
      }
      if (event.key === " " && passAction) {
        event.preventDefault();
        void submit(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [passAction, selectedCards, selectedAction, submit]);

  return {
    game, selected, selectedCards, selectedAction, suggestions, passAction,
    levelChoice, aiMode, busy, error, showRules,
    setLevelChoice, setAiMode, setError, setShowRules, setSelected,
    startGame, submit, toggleCard, selectSuggestion, showHint,
  };
}