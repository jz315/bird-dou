import type { GuandanState } from "../types";

const SEAT_NAMES = ["你", "左家", "对家", "右家"];

export function SeatBadge({
  seat,
  position,
  game,
}: {
  seat: number;
  position: "left" | "top" | "right";
  game: GuandanState;
}) {
  const finished = game.finishOrder.indexOf(seat);
  return (
    <div className={`gd-seat gd-seat-${position} ${game.currentPlayer === seat ? "active" : ""}`}>
      <div className="gd-avatar">{SEAT_NAMES[seat].slice(0, 1)}</div>
      <div>
        <strong>{SEAT_NAMES[seat]} {seat === 2 && <mark>搭档</mark>}</strong>
        <span>{finished >= 0 ? `第 ${finished + 1} 名` : `${game.cardsLeft[seat]} 张`}</span>
      </div>
      <i className="gd-card-back"><b>{game.cardsLeft[seat]}</b></i>
    </div>
  );
}

export { SEAT_NAMES };

