import type { CSSProperties } from "react";
import { cardView } from "../cards";
import type { GuandanRank } from "../types";

interface Props {
  id: number;
  level: GuandanRank;
  size?: "hand" | "played" | "mini";
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;

  /**
   * 手牌中的排列序号。
   * 序号越大，z-index 越大，因此右边的牌永远盖住左边的牌。
   */
  index?: number;

  style?: CSSProperties;
}

export function GuandanCard({
  id,
  level,
  size = "hand",
  selected = false,
  disabled = false,
  onClick,
  onDoubleClick,
  index = 0,
  style,
}: Props) {
  const card = cardView(id, level);

  const physicalLabel =
    `${card.label}${card.suit}，第${Math.floor(id / 54) + 1}副`;

  const cardClassName = [
    "gd-card",
    `gd-card-${size}`,
    card.red ? "red" : "black",
    card.joker ? "joker" : "",
    card.wild ? "wild" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const content = (
    <>
      <span className="gd-card-corner">
        <b>{card.label}</b>
        <i>{card.suit}</i>
      </span>

      <span className="gd-card-center">
        {card.joker ? (
          <>
            <em>{card.rankIndex === 13 ? "B" : "R"}</em>
            <small>JOKER</small>
          </>
        ) : (
          card.suit
        )}
      </span>

      {card.wild && <span className="gd-wild-tag">配</span>}
    </>
  );

  /*
   * 出牌区、迷你牌等不可点击的牌不需要槽位结构。
   */
  if (!onClick) {
    return (
      <span
        className={cardClassName}
        data-card-id={id}
        aria-label={physicalLabel}
        style={style}
      >
        {content}
      </span>
    );
  }

  /*
   * 手牌使用“窄槽位 + 完整牌面”的结构：
   *
   * button 只占一小段横向宽度；
   * 内部牌面仍然是完整宽度；
   * 后一个 button 的 z-index 永远更大；
   * selected 只能改变纵向位置，不能改变 z-index。
   */
  return (
    <button
      type="button"
      className={[
        "gd-card-hitbox",
        selected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      data-card-id={id}
      disabled={disabled}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      style={{
        ...style,

        /*
         * 使用真实的内联 zIndex，而不是 CSS calc。
         * 这样不会受到 CSS 自定义变量或旧规则干扰。
         */
        zIndex: index + 1,
      }}
      aria-pressed={selected}
      aria-label={`${physicalLabel}${card.wild ? "，逢人配" : ""}`}
    >
      <span
        className={cardClassName}
        aria-hidden="true"
      >
        {content}
      </span>
    </button>
  );
}