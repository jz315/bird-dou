import { useCallback, useEffect, useMemo, useState } from "react";
import { createGame, playAction } from "./api";
import type { AiMode, GameAction, GameState } from "./types";

const RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "小王", "大王"];
const SUITS = ["♠", "♥", "♣", "♦"];
const SEAT_NAMES = ["你", "北风", "南山"];
const PHASE_LABELS: Record<GameState["phase"], string> = {
  bidding: "叫地主",
  doubling: "加倍",
  card_play: "出牌",
  terminal: "本局结束",
};

type CardRef = { key: string; rank: number; copy: number };

function cardsFromCounts(counts: number[]): CardRef[] {
  return counts.flatMap((count, rank) =>
    Array.from({ length: count }, (_, copy) => ({ key: `${rank}-${copy}`, rank, copy })),
  );
}

function sameCounts(left: number[], right: number[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function roleLabel(state: GameState, seat: number) {
  if (state.landlord === null) return "等待身份";
  return state.landlord === seat ? "地主" : "农民";
}

function relativeOpponentSeats(humanSeat: number) {
  return [(humanSeat + 1) % 3, (humanSeat + 2) % 3] as const;
}

function App() {
  const [game, setGame] = useState<GameState | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aiMode, setAiMode] = useState("heuristic");
  const [showRules, setShowRules] = useState(false);
  const [hintIndex, setHintIndex] = useState(0);

  const startGame = useCallback(async (mode = aiMode) => {
    setBusy(true);
    setError(null);
    setSelected(new Set());
    try {
      const state = await createGame(mode);
      setGame(state);
      setAiMode(state.aiMode);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "开局失败");
    } finally {
      setBusy(false);
    }
  }, [aiMode]);

  useEffect(() => {
    void startGame("heuristic");
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handCards = useMemo(() => cardsFromCounts(game?.hand ?? Array(15).fill(0)), [game?.hand]);
  const selectedCounts = useMemo(() => {
    const counts = Array(15).fill(0) as number[];
    handCards.forEach((card) => {
      if (selected.has(card.key)) counts[card.rank] += 1;
    });
    return counts;
  }, [handCards, selected]);

  const selectedAction = useMemo(
    () => game?.legalActions.find((action) => action.kind !== "pass" && sameCounts(action.cards, selectedCounts)),
    [game?.legalActions, selectedCounts],
  );
  const passAction = game?.legalActions.find((action) => action.kind === "pass");
  const suggestions = useMemo(
    () => game?.legalActions.filter((action) => action.phase === "card_play" && action.kind !== "pass") ?? [],
    [game?.legalActions],
  );

  const submit = useCallback(async (action: GameAction | undefined) => {
    if (!game || !action || busy) return;
    setBusy(true);
    setError(null);
    try {
      const state = await playAction(game.gameId, action.index);
      setGame(state);
      setSelected(new Set());
      setHintIndex(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "动作提交失败");
    } finally {
      setBusy(false);
    }
  }, [busy, game]);

  const selectSuggestion = (action: GameAction) => {
    const next = new Set<string>();
    action.cards.forEach((count, rank) => {
      for (let copy = 0; copy < count; copy += 1) next.add(`${rank}-${copy}`);
    });
    setSelected(next);
  };

  const showHint = () => {
    if (!suggestions.length) return;
    selectSuggestion(suggestions[hintIndex % suggestions.length]);
    setHintIndex((value) => value + 1);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelected(new Set());
      if (event.key === "Enter" && selectedAction) void submit(selectedAction);
      if (event.key === " " && passAction) {
        event.preventDefault();
        void submit(passAction);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [passAction, selectedAction, submit]);

  const modes = game?.availableAiModes ?? [];
  const opponents = game ? relativeOpponentSeats(game.humanSeat) : ([1, 2] as const);
  const latestBySeat = (seat: number) => [...(game?.recentActions ?? [])].reverse().find((item) => item.actor === seat);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>B</span></div>
          <div>
            <div className="eyebrow">RESEARCH PLAYGROUND</div>
            <h1>BIRD-DOU</h1>
          </div>
        </div>
        <div className="top-stats" aria-label="牌局状态">
          <span><i className="status-dot" /> RUST CORE</span>
          <span>{game ? PHASE_LABELS[game.phase] : "连接中"}</span>
          <span className="multiplier">×{game?.multiplier ?? 1}</span>
        </div>
        <div className="top-actions">
          <button className="ghost-button" onClick={() => setShowRules(true)}>玩法</button>
          <button className="primary-small" onClick={() => void startGame()} disabled={busy}>新开一局</button>
        </div>
      </header>

      <section className="game-layout">
        <aside className="match-panel panel-glass">
          <div className="panel-heading">
            <span className="eyebrow">MATCH</span>
            <strong>对局信息</strong>
          </div>
          <dl className="match-facts">
            <div><dt>局号</dt><dd>#{game ? String(game.seed).slice(-6) : "------"}</dd></div>
            <div><dt>炸弹</dt><dd>{game?.bombCount ?? 0}</dd></div>
            <div><dt>身份</dt><dd>{game ? roleLabel(game, game.humanSeat) : "—"}</dd></div>
          </dl>
          <label className="mode-picker">
            <span>对手模式</span>
            <select value={aiMode} onChange={(event) => setAiMode(event.target.value)} disabled={busy}>
              {modes.length ? modes.map((mode) => <option key={mode.id} value={mode.id}>{mode.label}</option>) : <option>快速规则 AI</option>}
            </select>
          </label>
          <p className="mode-note">{modes.find((mode) => mode.id === aiMode)?.description ?? "信息安全的本地 AI 对手"}</p>
          <div className="bottom-cards">
            <span>地主底牌</span>
            <MiniCards counts={game?.bottomCards ?? Array(15).fill(0)} hidden={!game?.bottomCards.some(Boolean)} />
          </div>
          <div className="keyboard-help">
            <span><kbd>Enter</kbd> 出牌</span>
            <span><kbd>Space</kbd> 不出</span>
            <span><kbd>Esc</kbd> 清牌</span>
          </div>
        </aside>

        <section className="table-stage" aria-label="斗地主牌桌">
          <div className="felt-table">
            <div className="table-grain" />
            <PlayerBadge seat={opponents[0]} state={game} side="left" latest={latestBySeat(opponents[0])} />
            <PlayerBadge seat={opponents[1]} state={game} side="right" latest={latestBySeat(opponents[1])} />

            <div className="table-center">
              <div className="turn-orbit" data-active={game?.currentPlayer ?? -1}>
                <span className="orbit-ring" />
                <span className="orbit-label">{busy ? "AI 正在思考" : game?.terminal ? "牌局已结束" : game?.humanTurn ? "轮到你了" : "等待对手"}</span>
              </div>
              <div className="last-play">
                <span className="eyebrow">LAST PLAY</span>
                {game?.lastNonPass ? <PlayedCards action={game.lastNonPass} /> : <p>等待第一手牌</p>}
              </div>
            </div>

            <div className="human-zone">
              <div className="human-meta">
                <div className={`avatar human ${game?.currentPlayer === game?.humanSeat ? "active" : ""}`}>你</div>
                <div>
                  <strong>{game ? roleLabel(game, game.humanSeat) : "玩家"}</strong>
                  <span>{game?.cardsLeft[game.humanSeat] ?? 0} 张</span>
                </div>
              </div>
              <div className="hand" aria-label="你的手牌">
                {handCards.map((card) => (
                  <PlayingCard
                    key={card.key}
                    card={card}
                    selected={selected.has(card.key)}
                    disabled={!game?.humanTurn || game.phase !== "card_play" || busy}
                    onClick={() => setSelected((current) => {
                      const next = new Set(current);
                      if (next.has(card.key)) next.delete(card.key); else next.add(card.key);
                      return next;
                    })}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="control-deck panel-glass">
            {game?.humanTurn && game.phase !== "card_play" ? (
              <div className="phase-actions">
                <div><span className="eyebrow">YOUR DECISION</span><strong>{PHASE_LABELS[game.phase]}</strong></div>
                <div className="action-row">
                  {game.legalActions.map((action) => (
                    <button key={action.index} className={action.kind.includes("3") || action.kind === "double" ? "action-button emphasis" : "action-button"} onClick={() => void submit(action)} disabled={busy}>{action.label}</button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div className="suggestion-strip" aria-label="合法牌型建议">
                  <span className="eyebrow">LEGAL MOVES</span>
                  <div>
                    {suggestions.slice(0, 10).map((action) => (
                      <button key={action.index} onClick={() => selectSuggestion(action)} title={action.label}>{action.label}</button>
                    ))}
                    {suggestions.length > 10 && <span className="more-count">+{suggestions.length - 10}</span>}
                  </div>
                </div>
                <div className="play-controls">
                  <button className="ghost-button" onClick={() => setSelected(new Set())} disabled={!selected.size || busy}>重选</button>
                  <button className="ghost-button" onClick={showHint} disabled={!suggestions.length || busy}>提示</button>
                  <button className="pass-button" onClick={() => void submit(passAction)} disabled={!passAction || busy}>不出</button>
                  <button className="play-button" onClick={() => void submit(selectedAction)} disabled={!selectedAction || busy}>出牌 <span>↵</span></button>
                </div>
              </>
            )}
          </div>
        </section>

        <aside className="history-panel panel-glass">
          <div className="panel-heading">
            <span className="eyebrow">TIMELINE</span>
            <strong>出牌记录</strong>
          </div>
          <div className="history-list">
            {!game?.recentActions.length && <p className="empty-state">牌局刚刚开始</p>}
            {[...(game?.recentActions ?? [])].reverse().map((action) => (
              <div className="history-item" key={`${action.sequence}-${action.actor}`}>
                <span className={`seat-pip seat-${action.actor}`}>{action.actor === game?.humanSeat ? "你" : action.actor}</span>
                <div><strong>{SEAT_NAMES[action.actor ?? 0]}</strong><p>{action.label}</p></div>
                <small>{String((action.sequence ?? 0) + 1).padStart(2, "0")}</small>
              </div>
            ))}
          </div>
        </aside>
      </section>

      {busy && <div className="thinking-toast"><span /><span /><span /> AI 正在计算</div>}
      {error && <div className="error-toast" role="alert">{error}<button onClick={() => setError(null)}>×</button></div>}
      {game?.result && <ResultDialog game={game} onAgain={() => void startGame()} />}
      {showRules && <RulesDialog modes={modes} onClose={() => setShowRules(false)} />}
    </main>
  );
}

function PlayerBadge({ seat, state, side, latest }: { seat: number; state: GameState | null; side: "left" | "right"; latest?: GameAction }) {
  const active = state?.currentPlayer === seat;
  return (
    <div className={`opponent opponent-${side}`}>
      <div className={`avatar ${active ? "active" : ""}`}>{SEAT_NAMES[seat].slice(0, 1)}</div>
      <div className="opponent-copy">
        <strong>{SEAT_NAMES[seat]} {state?.landlord === seat && <span className="landlord-tag">地主</span>}</strong>
        <span>{state?.cardsLeft[seat] ?? 17} 张 · {state ? roleLabel(state, seat) : "AI"}</span>
      </div>
      <div className="card-back-stack" aria-label={`${state?.cardsLeft[seat] ?? 17} 张牌`}>
        <i /><i /><i /><b>{state?.cardsLeft[seat] ?? 17}</b>
      </div>
      {latest && <div className="opponent-play">{latest.label}</div>}
    </div>
  );
}

function PlayingCard({ card, selected, disabled, onClick }: { card: CardRef; selected: boolean; disabled: boolean; onClick: () => void }) {
  const joker = card.rank >= 13;
  const suit = joker ? (card.rank === 13 ? "JOKER" : "JOKER") : SUITS[card.copy % SUITS.length];
  const red = joker ? card.rank === 14 : suit === "♥" || suit === "♦";
  return (
    <button
      className={`playing-card ${red ? "red" : "black"} ${joker ? "joker" : ""} ${selected ? "selected" : ""}`}
      style={{ "--card-index": card.rank * 4 + card.copy } as React.CSSProperties}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      aria-label={`${RANKS[card.rank]}${joker ? "" : suit}`}
    >
      <span className="corner"><b>{RANKS[card.rank]}</b><i>{suit}</i></span>
      <span className="card-center">{joker ? <><em>{card.rank === 13 ? "B" : "R"}</em><small>JOKER</small></> : suit}</span>
    </button>
  );
}

function MiniCards({ counts, hidden }: { counts: number[]; hidden: boolean }) {
  const cards = cardsFromCounts(counts);
  if (hidden) return <div className="mini-cards hidden"><i /><i /><i /></div>;
  return <div className="mini-cards">{cards.map((card) => <i key={card.key}>{RANKS[card.rank]}</i>)}</div>;
}

function PlayedCards({ action }: { action: GameAction }) {
  if (action.kind === "pass") return <p className="pass-copy">不出</p>;
  return (
    <div className="played-group">
      <strong>{action.label.split(" · ")[0]}</strong>
      <div>{cardsFromCounts(action.cards).map((card) => <span key={card.key}>{RANKS[card.rank]}</span>)}</div>
    </div>
  );
}

function ResultDialog({ game, onAgain }: { game: GameState; onAgain: () => void }) {
  const won = game.result?.humanWon;
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="本局结果">
      <div className={`result-card ${won ? "win" : "loss"}`}>
        <span className="eyebrow">MATCH COMPLETE</span>
        <div className="result-symbol">{game.result?.allPass ? "↻" : won ? "胜" : "负"}</div>
        <h2>{game.result?.message}</h2>
        <p>最终倍数 ×{game.multiplier} · 共出现 {game.bombCount} 个炸弹</p>
        {!game.result?.allPass && <div className="score-grid">{game.result?.rawPayoff.map((score, seat) => <div key={seat}><span>{SEAT_NAMES[seat]}</span><strong className={score >= 0 ? "positive" : "negative"}>{score > 0 ? "+" : ""}{score}</strong></div>)}</div>}
        <button className="play-button" onClick={onAgain}>再来一局</button>
      </div>
    </div>
  );
}

function RulesDialog({ modes, onClose }: { modes: AiMode[]; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="玩法说明" onClick={onClose}>
      <div className="rules-card" onClick={(event) => event.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="关闭">×</button>
        <span className="eyebrow">HOW TO PLAY</span>
        <h2>完整斗地主，一局到底</h2>
        <div className="rules-grid">
          <section><b>01</b><h3>先叫地主</h3><p>你和两位 AI 依次叫分，最高叫分者获得三张底牌。</p></section>
          <section><b>02</b><h3>选择手牌</h3><p>点击手牌抬起，再按“出牌”；合法牌型建议可以一键选择。</p></section>
          <section><b>03</b><h3>农民合作</h3><p>地主一人对抗两位农民。炸弹、王炸、春天都会改变倍数。</p></section>
        </div>
        <div className="mode-explainer">
          {modes.map((mode) => <p key={mode.id}><strong>{mode.label}</strong>{mode.description}</p>)}
        </div>
      </div>
    </div>
  );
}

export default App;
