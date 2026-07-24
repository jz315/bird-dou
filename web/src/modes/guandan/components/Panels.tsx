import { MOVE_LABELS, levelLabel } from "../cards";
import type { GuandanState } from "../types";
import { SEAT_NAMES } from "./SeatBadge";

const LEVELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];

export function MatchPanel({
  game,
  levelChoice,
  setLevelChoice,
}: {
  game: GuandanState | null;
  levelChoice: number;
  setLevelChoice: (value: number) => void;
}) {
  return (
    <aside className="gd-match-panel panel-glass">
      <div className="panel-heading"><span className="eyebrow">MATCH</span><strong>本副信息</strong></div>
      <dl className="match-facts">
        <div><dt>局号</dt><dd>#{game ? game.gameId.slice(-6) : "------"}</dd></div>
        <div><dt>当前级牌</dt><dd>{game ? levelLabel(game.level) : "2"}</dd></div>
        <div><dt>搭档</dt><dd>对家 · 座位 2</dd></div>
      </dl>
      <label className="mode-picker">
        <span>新局级牌</span>
        <select value={levelChoice} onChange={(event) => setLevelChoice(Number(event.target.value))}>
          {LEVELS.map((label, index) => <option key={label} value={index}>{label}</option>)}
        </select>
      </label>
      <div className="gd-hierarchy">
        <span>牌力阶梯</span>
        <p>四王</p><i /> <p>六炸+</p><i /> <p>同花顺</p><i /> <p>五炸</p><i /> <p>四炸</p>
      </div>
      <div className="keyboard-help">
        <span><kbd>Enter</kbd> 出牌</span>
        <span><kbd>Space</kbd> 不出</span>
        <span><kbd>Esc</kbd> 清牌</span>
      </div>
    </aside>
  );
}

export function HistoryPanel({ game }: { game: GuandanState | null }) {
  return (
    <aside className="gd-history-panel panel-glass">
      <div className="panel-heading"><span className="eyebrow">TIMELINE</span><strong>行牌记录</strong></div>
      <div className="history-list">
        {!game?.recentActions.length && <p className="empty-state">等待第一手牌</p>}
        {[...(game?.recentActions ?? [])].reverse().map((event) => (
          <div className="history-item" key={event.sequence}>
            <span className={`seat-pip seat-${event.actor}`}>{event.actor === 0 ? "你" : event.actor}</span>
            <div><strong>{SEAT_NAMES[event.actor]}</strong><p>{MOVE_LABELS[event.kind] ?? event.kind} · {event.cards.length || "—"} 张</p></div>
            <small>{String(event.sequence + 1).padStart(2, "0")}</small>
          </div>
        ))}
      </div>
    </aside>
  );
}

