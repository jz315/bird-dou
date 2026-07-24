import { MOVE_LABELS, sortedCards } from "../cards";
import type { GuandanEvent, GuandanRank, GuandanTarget } from "../types";
import { GuandanCard } from "./GuandanCard";

export function PlayedMove({ move, level }: { move: GuandanTarget | GuandanEvent; level: GuandanRank; }) {
  if (move.kind === "pass") return <p className="gd-pass-copy">不出</p>;
  
  // 🔴 核心 UX：超过 5 张的牌型自动缩小展示
  const isLongMove = move.cards.length > 5;

  return (
    <div className="gd-played-move">
      <strong>{MOVE_LABELS[move.kind] ?? move.kind}</strong>
      <div className={isLongMove ? "gd-played-compact" : ""}>
        {sortedCards(move.cards, level).map((card) => (
          <GuandanCard key={card.id} id={card.id} level={level} size="played" />
        ))}
      </div>
    </div>
  );
}