import { levelLabel } from "../cards";
import type { GuandanState } from "../types";
import { SEAT_NAMES } from "./SeatBadge";

export function ResultDialog({
  game,
  onAgain,
}: {
  game: GuandanState;
  onAgain: () => void;
}) {
  const result = game.result;
  if (!result) return null;
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="本局结果">
      <div className={`result-card ${result.humanWon ? "win" : "loss"}`}>
        <span className="eyebrow">ROUND COMPLETE</span>
        <div className="result-symbol">{result.humanWon ? "胜" : "负"}</div>
        <h2>{result.humanWon ? "你和对家拿下这一副" : "对手组合赢得这一副"}</h2>
        <p>本轮打 {levelLabel(game.level)} · 胜方升 {result.levelAdvance} 级</p>
        <div className="gd-finish-grid">
          {result.finishOrder.map((seat, index) => (
            <div key={seat}><span>第 {index + 1} 名</span><strong>{SEAT_NAMES[seat]}</strong></div>
          ))}
        </div>
        <button className="play-button" onClick={onAgain}>再来一副</button>
      </div>
    </div>
  );
}

export function RulesDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="rules-card" onClick={(event) => event.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="关闭">×</button>
        <span className="eyebrow">GUANDAN · TWO DECKS</span>
        <h2>四人对家，两副牌掼蛋</h2>
        <div className="rules-grid">
          <section><b>01</b><h3>逢人配</h3><p>两张红桃级牌可代替除大小王外的任意牌，并可组成最多十张炸弹。</p></section>
          <section><b>02</b><h3>固定连续牌</h3><p>顺子五张、连对三对、钢板两组三张；不能继续延长或额外带牌。</p></section>
          <section><b>03</b><h3>搭档接风</h3><p>某家出完后其他人都不出，由他的对家获得下一轮出牌权。</p></section>
        </div>
        <p className="gd-rule-order">四王 ＞ 六张及以上炸弹 ＞ 同花顺 ＞ 五炸 ＞ 四炸 ＞ 普通牌型</p>
      </div>
    </div>
  );
}

