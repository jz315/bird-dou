import { useMemo } from "react";
import { MOVE_LABELS, sortedCards } from "../cards";
import type { GuandanAction, GuandanState } from "../types";
import { GuandanCard } from "./GuandanCard";
import { PlayedMove } from "./PlayedMove";
import { SeatBadge } from "./SeatBadge";

interface Props {
  game: GuandanState;
  selected: Set<number>;
  selectedCards: number[];
  selectedAction?: GuandanAction;
  suggestions: GuandanAction[];
  passAvailable: boolean;
  busy: boolean;

  toggleCard: (id: number) => void;
  selectSuggestion: (action: GuandanAction) => void;
  clearSelection: () => void;
  showHint: () => void;
  submit: (cards: number[] | null) => Promise<void>;
}

/**
 * 根据手牌数量决定牌与牌之间露出的宽度。
 *
 * 注意：
 * 这里控制的是每张牌的“槽位宽度”，不是负 margin。
 * 每张牌本身仍然保持完整尺寸，只是后面的牌覆盖前面的牌。
 */
function getHandDensityClass(cardCount: number): string {
  if (cardCount >= 25) return "gd-hand--very-dense";
  if (cardCount >= 21) return "gd-hand--dense";
  if (cardCount >= 17) return "gd-hand--medium";
  return "gd-hand--loose";
}

export function GuandanTable(props: Props) {
  const {
    game,
    selected,
    selectedCards,
    selectedAction,
    suggestions,
    passAvailable,
    busy,
    toggleCard,
    selectSuggestion,
    clearSelection,
    showHint,
    submit,
  } = props;

  const canPlay = selectedCards.length > 0 && Boolean(selectedAction);

  const selectionText = selectedAction
    ? MOVE_LABELS[selectedAction.kind] ?? selectedAction.kind
    : selectedCards.length
      ? "无效牌型"
      : "";

  /*
   * 只排序一次。
   * 渲染顺序就是最终层叠顺序：
   * 左边 index 小，右边 index 大。
   */
  const handCards = useMemo(
    () => sortedCards(game.hand, game.level),
    [game.hand, game.level],
  );

  const handDensityClass = getHandDensityClass(handCards.length);

  const handleCardDoubleClick = (cardId: number) => {
    if (!selected.has(cardId) || !canPlay) return;
    void submit(selectedCards);
  };

  return (
    <section
      className="gd-table-stage"
      aria-label="四人两副牌掼蛋牌桌"
    >
      <div className="gd-felt-table">
        <div className="table-grain" />

        <SeatBadge seat={1} position="left" game={game} />
        <SeatBadge seat={2} position="top" game={game} />
        <SeatBadge seat={3} position="right" game={game} />

        <div className="gd-table-center">
          <span className="gd-turn-label">
            {busy
              ? "三位 AI 正在行牌"
              : game.result
                ? "本副结束"
                : game.humanTurn
                  ? "轮到你了"
                  : `等待座位 ${game.currentPlayer}`}
          </span>

          <div className="gd-current-play">
            <span className="eyebrow">CURRENT TARGET</span>

            {game.target ? (
              <PlayedMove move={game.target} level={game.level} />
            ) : (
              <p>自由出牌</p>
            )}
          </div>
        </div>

        <div className="gd-human-zone">
          <div className="gd-human-meta">
            <div
              className={`gd-avatar ${game.humanTurn ? "active" : ""}`}
            >
              你
            </div>

            <div>
              <strong>你 · 与对家同队</strong>
              <span>{game.cardsLeft[0]} 张</span>
            </div>
          </div>

          <div
            className={`gd-hand ${handDensityClass}`}
            aria-label="你的实体手牌"
          >
            {handCards.map((card, index) => (
              <GuandanCard
                key={card.id}
                id={card.id}
                level={game.level}
                index={index}
                selected={selected.has(card.id)}
                disabled={!game.humanTurn || busy}
                onClick={() => toggleCard(card.id)}
                onDoubleClick={() => handleCardDoubleClick(card.id)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="gd-control-deck panel-glass">
        <div className="suggestion-strip">
          <span className="eyebrow">LEGAL MOVES</span>

          <div>
            {suggestions.slice(0, 12).map((action) => (
              <button
                key={action.index}
                type="button"
                onClick={() => selectSuggestion(action)}
              >
                {MOVE_LABELS[action.kind] ?? action.kind}
                {" · "}
                {action.totalCards}
              </button>
            ))}

            {suggestions.length > 12 && (
              <span className="more-count">
                +{suggestions.length - 12}
              </span>
            )}
          </div>
        </div>

        <div className="play-controls">
          <span
            className={[
              "gd-selection-kind",
              selectedCards.length > 0 && !selectedAction ? "invalid" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            {selectionText}
          </span>

          <button
            type="button"
            className="ghost-button"
            onClick={clearSelection}
            disabled={!selected.size || busy}
          >
            重选
          </button>

          <button
            type="button"
            className="ghost-button"
            onClick={showHint}
            disabled={!suggestions.length || busy}
          >
            提示
          </button>

          <button
            type="button"
            className="pass-button"
            onClick={() => void submit(null)}
            disabled={!passAvailable || busy}
          >
            不出
          </button>

          <button
            type="button"
            className="play-button"
            onClick={() => void submit(selectedCards)}
            disabled={!canPlay || busy}
          >
            出牌 <span>↗</span>
          </button>
        </div>
      </div>
    </section>
  );
}